## fineting 
llamafactory-cli train train_config.yaml

## merge stage1 model 
llamafactory-cli export merge_lora_dpo.yaml

## run model inference
llamafactory-cli api model_inference.yaml