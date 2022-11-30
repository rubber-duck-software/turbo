use anyhow::Result;
use indexmap::IndexMap;
use turbo_tasks_env::{
    CommandLineProcessEnvVc, EnvMapProcessEnvVc, EnvMapVc, FilterProcessEnvVc, MergeProcessEnvVc,
    ProcessEnvVc,
};
use turbo_tasks_fs::FileSystemPathVc;
use turbopack_env::TryDotenvProcessEnvVc;

use crate::config::NextConfigVc;

/// Loads a series of dotenv files according to the precedence rules set by
/// https://nextjs.org/docs/basic-features/environment-variables#environment-variable-load-order
#[turbo_tasks::function]
pub async fn load_env(
    project_path: FileSystemPathVc,
    config: NextConfigVc,
) -> Result<ProcessEnvVc> {
    let env = CommandLineProcessEnvVc::new().as_process_env();
    let node_env = env.read("NODE_ENV").await?;
    let node_env = node_env.as_deref().unwrap_or("development");

    let files = [
        Some(format!(".env.{node_env}.local")),
        if node_env == "test" {
            None
        } else {
            Some(".env.local".into())
        },
        Some(format!(".env.{node_env}")),
        Some(".env".into()),
    ]
    .into_iter()
    .flatten();

    let env = files.fold(env, |prior, f| {
        let path = project_path.join(&f);
        TryDotenvProcessEnvVc::new(prior, path).as_process_env()
    });

    let next_env = get_next_env(config);

    let env = MergeProcessEnvVc::new(env, next_env).as_process_env();

    Ok(env)
}

#[turbo_tasks::function]
pub async fn get_next_env(config: NextConfigVc) -> Result<ProcessEnvVc> {
    let config = &*config.await?;

    let mut map = IndexMap::new();

    if config.react_strict_mode.unwrap_or(false) {
        map.insert("__NEXT_STRICT_MODE".to_string(), "true".to_string());
    }

    if config.react_strict_mode.unwrap_or(true) {
        map.insert("__NEXT_STRICT_MODE_APP".to_string(), "true".to_string());
    }

    Ok(EnvMapProcessEnvVc::cell(EnvMapVc::cell(map)).into())
}

pub fn filter_for_client(env: ProcessEnvVc) -> ProcessEnvVc {
    MergeProcessEnvVc::new(
        FilterProcessEnvVc::new(env, "NEXT_PUBLIC_".to_string()).into(),
        FilterProcessEnvVc::new(env, "__NEXT_".to_string()).into(),
    )
    .into()
}
