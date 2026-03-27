memory-eval run \
    --task healthbench \
    --model-config azure-gpt-4.1

# memory-eval evaluate \
#     --task healthbench \
#     --model-config azure-gpt-4.1 \
#     --grader-config azure-gpt-4.1

memory-eval evaluate \
    --task healthbench \
    --model-config azure-gpt-4.1 \
    --grader-config qwen3.5-122b-a10b