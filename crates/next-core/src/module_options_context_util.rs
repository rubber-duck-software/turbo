use anyhow::Result;
use turbopack::module_options::ModuleOptionsContextVc;

#[turbo_tasks::function]
pub async fn add_next_font_transform(
    module_options_context: ModuleOptionsContextVc,
) -> Result<ModuleOptionsContextVc> {
    #[cfg(not(feature = "next-font"))]
    return Ok(module_options_context);

    #[cfg(feature = "next-font")]
    {
        use turbopack::module_options::{ModuleRule, ModuleRuleCondition, ModuleRuleEffect};
        use turbopack_ecmascript::{EcmascriptInputTransform, EcmascriptInputTransformsVc};

        let mut module_options_context = module_options_context.await?.clone_value();
        module_options_context.custom_rules.push(ModuleRule::new(
            // TODO: Only match in pages (not pages/api), app/, etc.
            ModuleRuleCondition::any(vec![
                ModuleRuleCondition::ResourcePathEndsWith(".js".to_string()),
                ModuleRuleCondition::ResourcePathEndsWith(".jsx".to_string()),
                ModuleRuleCondition::ResourcePathEndsWith(".ts".to_string()),
                ModuleRuleCondition::ResourcePathEndsWith(".tsx".to_string()),
            ]),
            vec![ModuleRuleEffect::AddEcmascriptTransforms(
                EcmascriptInputTransformsVc::cell(vec![EcmascriptInputTransform::NextJsFont]),
            )],
        ));
        Ok(module_options_context.cell())
    }
}
