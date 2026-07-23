// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 expert-pack structural tests (GPU-free): the `present` gate and the
//! sizing key-lists derived from an audit. The device pack + shape audit run on
//! hardware / against a real WeightStore.

use std::collections::BTreeMap;

use super::{ExpertMap, RouterMap, key_lists, present};
use crate::lora::ExpertProj;

fn empty() -> (RouterMap, ExpertMap) {
    (BTreeMap::new(), BTreeMap::new())
}

#[test]
fn present_is_false_only_when_both_empty() {
    let (r, e) = empty();
    assert!(!present(&r, &e));

    let (mut r, e) = empty();
    r.insert(3, [Some("a".into()), Some("b".into())]);
    assert!(present(&r, &e));

    let (r, mut e) = empty();
    e.insert((7, 2, ExpertProj::Down), [None, None]);
    assert!(present(&r, &e));
}

#[test]
fn key_lists_projects_layers_and_projs() {
    let mut router: RouterMap = BTreeMap::new();
    router.insert(3, [None, None]);
    router.insert(7, [None, None]);
    let mut experts: ExpertMap = BTreeMap::new();
    experts.insert((7, 0, ExpertProj::Gate), [None, None]);
    experts.insert((7, 5, ExpertProj::Down), [None, None]);

    let (ek, rl) = key_lists(&router, &experts);
    assert_eq!(ek, vec![(7, ExpertProj::Gate), (7, ExpertProj::Down)]);
    assert_eq!(rl, vec![3, 7]);
}
