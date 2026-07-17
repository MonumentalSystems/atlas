// SPDX-License-Identifier: AGPL-3.0-only

//! Routed-expert worklist contract for small-batch MoE decode.
//!
//! Decode routes are flattened token-major (`route = token * top_k + slot`).
//! The persistent CUDA worker will consume the same expert-major groups this
//! module describes.  Groups contain at most eight *real* rows: unlike
//! Marlin's block-M metadata, no rows are padded merely to fill a tile.

/// Maximum number of routed rows owned by one decode worker group.
pub(crate) const MAX_DECODE_GROUP_ROWS: usize = 8;

/// One contiguous expert-major slice of the decode worklist.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodeExpertGroup {
    /// Routed expert selected for every row in this group.
    pub expert_id: u32,
    /// Index into [`DecodeWorklist::sorted_routes`].
    pub route_start: u32,
    /// Number of live (not padded) rows in this group.
    pub rows: u8,
}

/// Device-neutral routing metadata for the persistent grouped decode kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodeWorklist {
    /// Groups ordered by expert, then by the original token-major route.
    pub groups: Vec<DecodeExpertGroup>,
    /// Maps an expert-major worklist row back to its token-major route.
    pub sorted_routes: Vec<u32>,
    /// Reverse map used by the final weighted reduction/scatter.
    pub route_to_sorted: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeWorklistError {
    InvalidExpert { route: usize, expert: u32 },
}

impl DecodeWorklist {
    /// Builds stable expert-major groups from flattened top-k expert ids.
    ///
    /// The GPU builder must preserve these semantics exactly. In particular,
    /// rows that route to the same expert are grouped before any eight-row
    /// limit is applied, and no synthetic padded route is emitted.
    pub fn build(topk_ids: &[u32], num_experts: u32) -> Result<Self, DecodeWorklistError> {
        let mut routes_by_expert = vec![Vec::new(); num_experts as usize];
        for (route, &expert) in topk_ids.iter().enumerate() {
            let Some(bucket) = routes_by_expert.get_mut(expert as usize) else {
                return Err(DecodeWorklistError::InvalidExpert { route, expert });
            };
            bucket.push(route as u32);
        }

        let mut groups = Vec::new();
        let mut sorted_routes = Vec::with_capacity(topk_ids.len());
        let mut route_to_sorted = vec![0; topk_ids.len()];
        for (expert, routes) in routes_by_expert.into_iter().enumerate() {
            for chunk in routes.chunks(MAX_DECODE_GROUP_ROWS) {
                let route_start = sorted_routes.len() as u32;
                for &route in chunk {
                    route_to_sorted[route as usize] = sorted_routes.len() as u32;
                    sorted_routes.push(route);
                }
                groups.push(DecodeExpertGroup {
                    expert_id: expert as u32,
                    route_start,
                    rows: chunk.len() as u8,
                });
            }
        }

        Ok(Self {
            groups,
            sorted_routes,
            route_to_sorted,
        })
    }

    /// Total live rows. This is exactly the flattened route count.
    pub fn live_rows(&self) -> usize {
        self.sorted_routes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_routes_stably_without_padding() {
        let plan = DecodeWorklist::build(&[3, 1, 3, 1, 2], 4).unwrap();
        assert_eq!(plan.live_rows(), 5);
        assert_eq!(plan.sorted_routes, vec![1, 3, 4, 0, 2]);
        assert_eq!(
            plan.groups,
            vec![
                DecodeExpertGroup {
                    expert_id: 1,
                    route_start: 0,
                    rows: 2
                },
                DecodeExpertGroup {
                    expert_id: 2,
                    route_start: 2,
                    rows: 1
                },
                DecodeExpertGroup {
                    expert_id: 3,
                    route_start: 3,
                    rows: 2
                },
            ]
        );
        assert_eq!(plan.route_to_sorted, vec![3, 0, 4, 1, 2]);
    }

    #[test]
    fn splits_only_a_hot_expert_at_eight_live_rows() {
        let plan = DecodeWorklist::build(&[2; 17], 4).unwrap();
        assert_eq!(plan.live_rows(), 17);
        assert_eq!(plan.sorted_routes, (0..17).collect::<Vec<_>>());
        assert_eq!(
            plan.groups,
            vec![
                DecodeExpertGroup {
                    expert_id: 2,
                    route_start: 0,
                    rows: 8
                },
                DecodeExpertGroup {
                    expert_id: 2,
                    route_start: 8,
                    rows: 8
                },
                DecodeExpertGroup {
                    expert_id: 2,
                    route_start: 16,
                    rows: 1
                },
            ]
        );
    }

    #[test]
    fn route_and_sorted_maps_round_trip() {
        let ids = [5, 0, 5, 2, 0, 5, 2, 5];
        let plan = DecodeWorklist::build(&ids, 6).unwrap();
        for (sorted, &route) in plan.sorted_routes.iter().enumerate() {
            assert_eq!(plan.route_to_sorted[route as usize], sorted as u32);
        }
    }

    #[test]
    fn rejects_out_of_range_experts() {
        assert_eq!(
            DecodeWorklist::build(&[0, 4], 4),
            Err(DecodeWorklistError::InvalidExpert {
                route: 1,
                expert: 4,
            })
        );
    }

    #[test]
    fn c8_sparse_routes_remain_unpadded() {
        let ids: Vec<u32> = (0..64).map(|route| route % 57).collect();
        let plan = DecodeWorklist::build(&ids, 256).unwrap();
        assert_eq!(plan.live_rows(), 64);
        assert_eq!(plan.groups.len(), 57);
        assert!(plan.groups.iter().all(|group| group.rows <= 2));
    }
}
