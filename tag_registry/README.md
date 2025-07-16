This directory holds canonical YAML definitions for every tag used by the project.  

Each file inside this folder (except `registry_index.yaml` and this README) defines **one** tag and **must** conform to the JSON Schema located at `schemas/tag_definition.schema.json`.

File-naming convention: `<tag_name>.yaml`

Required companion files:
• `registry_index.yaml` – machine-generated list of all tags, their versions, and the YAML file that defines them.  
• Schema validation script: `scripts/validate_tags.py` (run via pre-commit / CI).

Run validation locally:
```bash
python scripts/validate_tags.py
```
Any schema violations or index inconsistencies will exit with a non-zero status.