// SPDX-License-Identifier: AGPL-3.0-only
//
// Minimal, dependency-light safetensors reader for the offline expert builder.
//
// We only need three things from a checkpoint: the tensor directory (name ->
// dtype/shape/byte-range), and the raw bytes of a handful of named tensors per
// expert. The full `safetensors` crate (and an mmap of a 22 GB file) is
// overkill; the format is a fixed 8-byte little-endian header length, a JSON
// header, then the tensor data blob. We parse the header once and `pread` only
// the byte ranges we actually use, so peak RSS stays tiny regardless of
// checkpoint size.
//
// This is a *reader for the builder*, deliberately separate from the runtime's
// `spark_runtime::weights::SafetensorsLoader` (which is CUDA-coupled): the
// builder must run on any host, GPU or not.

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// One tensor's metadata as recorded in the safetensors JSON header.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte range within the data blob (i.e. relative to `data_start`).
    pub begin: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn nbytes(&self) -> u64 {
        self.end - self.begin
    }
    pub fn rows(&self) -> u64 {
        self.shape.first().copied().unwrap_or(0)
    }
    pub fn cols(&self) -> u64 {
        self.shape.get(1).copied().unwrap_or(0)
    }
}

/// A parsed safetensors file: the header directory plus an open fd positioned to
/// `pread` tensor ranges on demand.
pub struct SafeTensors {
    file: File,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("open safetensors {}", path.display()))?;
        let mut lenbuf = [0u8; 8];
        file.read_exact_at(&mut lenbuf, 0)
            .context("read safetensors header length")?;
        let header_len = u64::from_le_bytes(lenbuf);
        // Guard against a corrupt/huge length before allocating.
        if header_len == 0 || header_len > 512 * 1024 * 1024 {
            bail!("implausible safetensors header length: {header_len}");
        }
        let mut hdr = vec![0u8; header_len as usize];
        file.read_exact_at(&mut hdr, 8)
            .context("read safetensors header json")?;
        let json: serde_json::Value =
            serde_json::from_slice(&hdr).context("parse safetensors header json")?;
        let obj = json
            .as_object()
            .context("safetensors header is not a json object")?;

        let mut tensors = BTreeMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let m = meta
                .as_object()
                .with_context(|| format!("tensor {name} metadata not an object"))?;
            let dtype = m
                .get("dtype")
                .and_then(|v| v.as_str())
                .with_context(|| format!("tensor {name} missing dtype"))?
                .to_string();
            let shape = m
                .get("shape")
                .and_then(|v| v.as_array())
                .with_context(|| format!("tensor {name} missing shape"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0))
                .collect::<Vec<_>>();
            let offs = m
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .with_context(|| format!("tensor {name} missing data_offsets"))?;
            if offs.len() != 2 {
                bail!("tensor {name} data_offsets not [begin,end]");
            }
            let begin = offs[0].as_u64().unwrap_or(0);
            let end = offs[1].as_u64().unwrap_or(0);
            if end < begin {
                bail!("tensor {name} has end < begin");
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            );
        }
        Ok(Self {
            file,
            data_start: 8 + header_len,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Read a tensor's raw bytes into a fresh `Vec` (exactly `nbytes()` long).
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        let mut buf = vec![0u8; info.nbytes() as usize];
        self.file
            .read_exact_at(&mut buf, self.data_start + info.begin)
            .with_context(|| format!("read tensor {name}"))?;
        Ok(buf)
    }

    /// Read a scalar F32 tensor (shape `[1]`).
    pub fn read_f32_scalar(&self, name: &str) -> Result<f32> {
        let info = self
            .get(name)
            .with_context(|| format!("scalar {name} not found"))?;
        if info.dtype != "F32" || info.nbytes() != 4 {
            bail!(
                "scalar {name}: expected F32[1] (4 bytes), got {} {} bytes",
                info.dtype,
                info.nbytes()
            );
        }
        let bytes = self.read_tensor(name)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}
