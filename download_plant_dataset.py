from datasets import load_dataset
from collections import Counter

dataset = load_dataset(
    "Project-AgML/DIMPSAR_medicinal_plant_classification"
)

train = dataset["train"]

print("Total images:", len(train))

print("\nFeatures:")
print(train.features)

print("\nNumber of classes:", len(train.features["label"].names))

print("\nClasses:")
for i, name in enumerate(train.features["label"].names):
    print(i, "->", name)

print("\nImages per class:")
counts = Counter(train["label"])

for label_id, count in sorted(counts.items()):
    name = train.features["label"].names[label_id]
    print(f"{name}: {count}")