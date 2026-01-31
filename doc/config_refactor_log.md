Summary of Changes:

   1. ✅ config.yaml - Added 2 control variables at top
      - active_train_profile: "origabcde"
      - active_score_dataset: "f"

   2. ✅ config.yaml - Updated winner section
      - Uses {active_train_profile} templates
      - Removed 6 commented lines

   3. ✅ config.yaml - Updated winnerscore section
      - Uses {active_train_profile} and {active_score_dataset} templates
      - Removed 30+ commented lines

   4. ✅ service/env_config.py - Added template resolution
      - _resolve_template() function
      - Integrated into get() method

   5. ✅ Documentation created
      - doc/config_quick_fix_guide.md (full usage guide)

   Files Modified:
     - config.yaml (backup: config.yaml.backup)
     - service/env_config.py
     - doc/config_quick_fix_guide.md (new)

   How to Use:
     - Change 2 lines at top of config.yaml
     - Everything else syncs automatically
     - See doc/config_quick_fix_guide.md for examples

   Next Steps:
     1. Test with your normal workflow
     2. Add dataset g in 2 weeks (3-step process)
     3. Enjoy cleaner config! 🚀