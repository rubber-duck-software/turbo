use anyhow::Result;
use indexmap::IndexMap;

use crate::{EnvMapVc, ProcessEnv, ProcessEnvVc};

/// Merges two environments, `merge` will be merged into `base` potentially
/// overriding values in `base`.
#[turbo_tasks::value]
pub struct MergeProcessEnv {
    base: ProcessEnvVc,
    merge: ProcessEnvVc,
}

#[turbo_tasks::value_impl]
impl MergeProcessEnvVc {
    #[turbo_tasks::function]
    pub fn new(base: ProcessEnvVc, merge: ProcessEnvVc) -> Self {
        MergeProcessEnv { base, merge }.cell()
    }
}

#[turbo_tasks::value_impl]
impl ProcessEnv for MergeProcessEnv {
    #[turbo_tasks::function]
    async fn read_all(&self) -> Result<EnvMapVc> {
        let base = &*self.base.read_all().await?;
        let merge = &*self.merge.read_all().await?;

        // adding the lengths assuming there won't be any/many conflicts
        let mut merged = IndexMap::with_capacity(base.len() + merge.len());
        merged.extend(base.clone().into_iter());
        merged.extend(merge.clone().into_iter());

        Ok(EnvMapVc::cell(merged))
    }
}
