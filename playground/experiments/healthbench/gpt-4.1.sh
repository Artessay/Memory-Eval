memory-eval run \
    --task healthbench \
    --model-backend azure \
    --model-name gpt-4.1

memory-eval evaluate \
    --task healthbench \
    --model-backend azure \
    --model-name gpt-4.1 \
    --grader-backend azure \
    --grader-model gpt-4.1