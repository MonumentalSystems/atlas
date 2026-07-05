// SPDX-License-Identifier: AGPL-3.0-only
//
// Checkpoint abstraction over one-or-many safetensors shards.
//
// Single-file checkpoints (AEON-7, AgentWorld) carry every tensor in one
// `model.safetensors`. Sharded checkpoints (Sehyo 122B, and the 397B target)
// ship `model.safetensors.index.json` with a `weight_map` of
// `tensor_name -> shard_file`. Both resolve to the same interface: given a
// tensor name, read its raw bytes / scalar value from whichever shard owns it.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::safetensors_min::{SafeTensors, TensorInfo};

pub struct Checkpoint {
    /// tensor name -> shard handle owning it.
    owner: HashMap<String, Rc<SafeTensors>>,
}

impl Checkpoint {
    /// Open a checkpoint directory, resolving single-file or sharded layout.
    pub fn open(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let single = dir.join("model.safetensors");
        let mut owner: HashMap<String, Rc<SafeTensors>> = HashMap::new();

        if index.exists() {
            let json = std::fs::read_to_string(&index)
                .with_context(|| format!("read {}", index.display()))?;
            let v: serde_json::Value =
                serde_json::from_str(&json).with_context(|| format!("parse {}", index.display()))?;
            let wm = v
                .get("weight_map")
                .and_then(|w| w.as_object())
                .context("index.json missing weight_map")?;
            // Open each distinct shard once.
            let mut shards: HashMap<String, Rc<SafeTensors>> = HashMap::new();
            for (name, shard) in wm {
                let file = shard.as_str().context("weight_map value not a string")?;
                let handle = match shards.get(file) {
                    Some(h) => h.clone(),
                    None => {
                        let p: PathBuf = dir.join(file);
                        let h = Rc::new(SafeTensors::open(&p)?);
                        shards.insert(file.to_string(), h.clone());
                        h
                    }
                };
                owner.insert(name.clone(), handle);
            }
        } else if single.exists() {
            let h = Rc::new(SafeTensors::open(&single)?);
            for name in h.names() {
                owner.insert(name.clone(), h.clone());
            }
        } else {
            bail!(
                "no model.safetensors or model.safetensors.index.json in {}",
                dir.display()
            );
        }
        Ok(Self { owner })
    }

    pub fn has(&self, name: &str) -> bool {
        self.owner.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.owner.get(name).and_then(|s| s.get(name))
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        self.owner
            .get(name)
            .with_context(|| format!("tensor {name} not in any shard"))?
            .read_tensor(name)
    }

    pub fn read_f32_scalar(&self, name: &str) -> Result<f32> {
        self.owner
            .get(name)
            .with_context(|| format!("scalar {name} not in any shard"))?
            .read_f32_scalar(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.owner.keys()
    }
}
