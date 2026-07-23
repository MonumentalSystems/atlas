// SPDX-License-Identifier: AGPL-3.0-only

//! FFI-surface tests. In CI there is no `libatlasb12x.so`, so `lib()` is `None` and the
//! whole surface must degrade to "unavailable" — never panic, never claim a capacity.

#[test]
fn unavailable_without_lib() {
    // No shim library on the test host => dlopen fails => lib() is None.
    assert!(
        !super::available(),
        "b12x must be unavailable without the shim lib"
    );
    assert_eq!(
        super::max_tokens(),
        None,
        "no capacity without the shim lib"
    );
}
