# memory-eval run \
#     --task healthbench \
#     --subset hard \
#     --model-config qwen2.5-1.5b 

memory-eval evaluate \
    --task healthbench \
    --subset hard \
    --model-config qwen2.5-1.5b \
    --grader-config qwen3.5-122b-a10b
