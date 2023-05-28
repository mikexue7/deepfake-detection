import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from load_data import return_data

img = Image.open('/users/jacobmejia/Downloads/fake_7.jpg')


# fig = plt.figure()
# plt.imshow(img)

# # resize to imagenet size 
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # add batch dim
# print(x.shape)

# patch_size = 16 # 16 pixels
# pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

# NOTE:CHANGED THE EMBED SIZE FROM 768 TO 256
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 256, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
                
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 256, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
        
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 512,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

def eval_model(model, loader, device, preprocess_fn=None, postprocess_fn=None):
    num_correct, num_samples, total_loss = 0, 0, 0
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            batch_size = x.shape[0]
            x = x.to(device=device)
            y = y.to(device=device)
            if preprocess_fn:
                x, y = preprocess_fn(x, y)
            scores = model(x)
            probs = torch.sigmoid(scores)
            if postprocess_fn:
                probs, y = postprocess_fn(probs, y, batch_size)
            preds = torch.round(probs)
            num_correct += (preds == y).sum()
            num_samples += batch_size
            total_loss += F.binary_cross_entropy(probs, y, reduction='sum')

            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
        acc = float(num_correct) / num_samples
        log_loss = total_loss.item() / num_samples
    return acc, log_loss, y_true, y_pred            

model = ViT()
out = model(x)
print(out)
print(summary(ViT(), (3, 224, 224), device='cpu'))
#loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used: ", device)
model.to(device)

# # Training loop
# num_epochs = 10
# print("Loading in Data")
# train_dataloader = return_data()
# print("Successfully Loaded In Data")

# for epoch in range(num_epochs):
#     print("Beginning Training")
#     running_loss = 0.0
#     for images, labels in train_dataloader:
#         images = images.to(device)
#         labels = labels.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(images)
#         # Compute the loss
#         #loss = loss_fn(torch.sigmoid(outputs), labels)
#         loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels, reduction='mean')
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Update the running loss
#         running_loss += loss.item()

#     # Print the average loss for the epoch
#     epoch_loss = running_loss / len(train_dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#     acc, log_loss, _, _ = eval_model(model, train_dataloader, device, None, None)
#     print('Training accuracy = ' , 100 * acc) # official log loss score


