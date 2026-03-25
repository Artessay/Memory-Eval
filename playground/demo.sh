memory-eval run \
    --task healthbench \
    --model-backend azure \
    --model-name gpt-4o \
    --limit 10

memory-eval evaluate \
    --task healthbench \
    --model-backend azure \
    --model-name gpt-4o \
    --limit 10 \
    --grader-backend azure \
    --grader-model gpt-4o