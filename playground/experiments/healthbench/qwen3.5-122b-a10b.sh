memory-eval run \
    --task healthbench \
    --model-config qwen3.5-122b-a10b

memory-eval evaluate \
    --task healthbench \
    --model-config qwen3.5-122b-a10b \
    --grader-config azure-gpt-4.1
