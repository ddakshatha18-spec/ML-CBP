from dataset import LUTSegDataset


root = "data/LUTSeg"

train_dataset = LUTSegDataset(
    root,
    split="train"
)

val_dataset = LUTSegDataset(
    root,
    split="val"
)

print("\nDataset sizes:")
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))

# Load the first training sample
image, mask = train_dataset[0]

print("\nFirst sample:")
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)

print("Mask classes:", sorted(mask.unique().tolist()))