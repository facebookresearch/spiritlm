# STSP Evaluation 

```python spiritlm/eval/eval_stsp.py --ref_file $REF_FILE --pred_file $pred_file```


e.g. 

```
python spiritlm/eval/eval_stsp.py\
 --ref_file ./data/stsp_data/records_emov_demo.jsonl \
 --pred_file ./data/stsp_data/emov_demo_pred.jsonl

> Accuracy: 100.00% for predictions ./data/stsp_data/emov_demo_pred.jsonl
```