use serde::{Deserialize, Serialize};
use turbo_tasks::trace::TraceRawVcs;

/// TODO: read complete next config directly
#[turbo_tasks::value(shared)]
pub struct NextConfig {
    pub react_strict_mode: Option<bool>,

    pub experimental: ExperimentalConfig,
}

#[derive(Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Serialize, Deserialize, TraceRawVcs)]
pub struct ExperimentalConfig {
    pub server_components_external_packages: Vec<String>,
}
