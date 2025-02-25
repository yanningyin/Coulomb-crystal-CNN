import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from skimage.util import random_noise
import numpy as np
import os
import glob
from datetime import datetime
import time
from preprocessing import load_data
from CLI import CLI
import time


# Set seed for random number generator
seed = 42  # Choose any seed value
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # If using CUDA
torch.backends.cudnn.deterministic = True  # If using CUDA and cuDNN


class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.label_mapping = self.create_label_mapping(labels)
        self.binarize = False
        self.add_noise = True

    def create_label_mapping(self, labels):
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def binarize_image(self, image, threshold=0.8):
        image_tensor = transforms.ToTensor()(image)
        binarized_image_tensor = (image_tensor > threshold).float()
        binary_image = transforms.ToPILImage()(binarized_image_tensor) 
        return binary_image

    def add_noise_to_image(self, image, noise_mode='gaussian'):
        image_array = np.array(image)
        noisy_image = random_noise(image_array, mode=noise_mode, mean=0, var=0.05)
        noisy_image = (255 * noisy_image).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_image)
        return noisy_image
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        if convert_to_rgb:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(img_path).convert('L')
        
        if self.binarize:
            image = self.binarize_image(image)
        
        if self.add_noise:
            image = self.add_noise_to_image(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        new_label = self.label_mapping[label]

        return image, new_label


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.i = 3
        if not convert_to_rgb:
            self.i = 1
        self.features = nn.Sequential(
            nn.Conv2d(self.i, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.classifier(x)
        return x


"""
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling layer after Conv1
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling layer after Conv2
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

        # Calculate your flattened input size based on the previous layer's output
        self.fc1 = nn.Linear(in_features= 27 * 27 * 128 , out_features=4096)  

        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # Applying Max Pooling after Conv1
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # Applying Max Pooling after Conv2
        x = self.norm2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
"""
        

cli = CLI()
input_image_dir = cli.args.input_image_dir
output_file_dir = cli.args.output_file_dir
mode = cli.args.mode
test_model_path = cli.args.test_model_path
label_to_classify = cli.args.label
range_n = cli.args.range_N
range_t = cli.args.range_T
class_range_n = cli.args.class_range_N
class_range_t = cli.args.class_range_T
check_image = cli.args.check_image
model = cli.args.model
use_pretrained_weights = cli.args.use_pretrained_weights
convert_to_rgb = cli.args.convert_to_rgb

if convert_to_rgb:
        transforms_normalize_mean = [0.485, 0.456, 0.406]
        transforms_normalize_std = [0.229, 0.224, 0.225]
else:
    transforms_normalize_mean = [0.5]
    transforms_normalize_std = [0.5]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random scaling
    #transforms.RandomRotation(degrees=30),  # Random rotation
    
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=90),
    transforms.ToTensor(),
    transforms.Normalize(mean=transforms_normalize_mean, std=transforms_normalize_std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=transforms_normalize_mean, std=transforms_normalize_std)
])

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


if mode == 'test':

    # find the images
    if type(input_image_dir) == str:
        image_path = os.path.join(os.path.dirname(__file__), input_image_dir)
        images= glob.glob(os.path.join(image_path, '*.tif'))
    elif type(input_image_dir) == list:
        images= []
        for d in input_image_dir:
            image_path = os.path.join(os.path.dirname(__file__), d)
            images += glob.glob(os.path.join(image_path, '*.tif'))
    else:
        print("Error: input_image_dir must be a string or a list!")
    
    print(images)
    model_basename = os.path.basename(test_model_path)
    parts = model_basename.split("_")
    class_mapping = {idx:label for idx, label in enumerate(range(int(parts[1]), 1+int(parts[2]), int(parts[3])))}
    
    verbose = True
    if verbose:
        print("---"*30)
        print("Class Mapping:", class_mapping)
        print("---"*30)

    net = torch.load(test_model_path, map_location=device)
    #print(net)
    net.eval()

    for image in images:
        start_time = time.perf_counter()
        if convert_to_rgb:
            img = Image.open(image).convert('RGB')
        else:
            img = Image.open(image).convert('L')

        input_tensor = test_transform(img)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():
            output = net(input_batch)
        _, predicted = torch.max(output, 1)
        used_time = time.perf_counter() - start_time
        print(f"Image: {os.path.basename(image)}, predicted class: {class_mapping[predicted.item()]}, time used: {used_time}")
else:
    print(f"Classifying the label {label_to_classify}")
    print(f"Range for N is {range_n}, for T is {range_t}")
    print(f"Checking image set to {check_image}")
    print(f"Model is set to {model}(pretrained={use_pretrained_weights})")
    print(f"Conversion of images to RGB mode is {convert_to_rgb}")
            

    img_paths_list, labels_n, _, labels_t = load_data(dir_name=input_image_dir,
                                                    range_num_ions=range_n, range_avg_temp=range_t,
                                                    check_image=check_image)
    if label_to_classify.lower() == 'n':
        labels_list = labels_n
        a, b = class_range_n[0]-1, class_range_n[-1]+1
        labels_list = [a if l <= a else b if l >= b else l for l in labels_list]
        step = class_range_n.step     
    elif label_to_classify.lower() == 't':
        labels_list = labels_t
        a, b = class_range_t[0]-1, class_range_t[-1]+1
        labels_list = [a if l <= a else b if l >= b else l for l in labels_list]
        step = class_range_t.step
    else:
        print("Classifying both N and T is not yet implemented")
        exit()

    # Set path to save log file and trained model
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(output_file_dir, f'{label_to_classify.lower()}_{min(labels_list)}_{max(labels_list)}_{step}_{model}_log_{datetime_str}.txt')
    weights_file = os.path.join(output_file_dir, f'{label_to_classify.lower()}_{min(labels_list)}_{max(labels_list)}_{step}_{model}_model_{datetime_str}.pth')

    print("trained model will be saved in ", weights_file)

    f = open(log_file, 'w')
    for arg in vars(cli.args):
        f.write(f'{arg}: {getattr(cli.args, arg)}\n')
    f.write(f"trained model saved in: {weights_file}")
    f.write(f'\nepoch\tloss\taccuracy(%)\n')


    dataset = CustomDataset(img_paths_list, labels_list, transform=train_transform)

    verbose = True
    if verbose:
        print('number of original labels n:', len(labels_n))
        print('number of original labels t:', len(labels_t))
        print("number of labels: ", len(labels_list))
        print(labels_list)
        print("---"*30)
        print("Label Mapping:", dataset.label_mapping)
        print("---"*30)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Apply the test_transform to the test dataset
    test_dataset.dataset = CustomDataset(test_dataset.dataset.img_paths, test_dataset.dataset.labels,
                                        transform=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Use a pre-trained ResNet-18 model
    num_classes = len(set(labels_list))
    print("number of classes:", num_classes)

    if model == "custom":
        net = CustomCNN(num_classes)
    else:
        net = eval(f"models.{model}(pretrained={use_pretrained_weights})")
        
        # Modify the first convolutional layer of the model to accept single-channel input:
        if not convert_to_rgb:
            if model == "alexnet":
                net.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
            elif model == "vgg16":
                net.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            elif "resnet" in model:
                net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            elif "mnasnet" in model:
                net.layers[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Modify the last layer to match the number of classes in your dataset
        if model in ["alexnet", "vgg16"]:
            net.classifier[6] = torch.nn.Linear(net.classifier[6].in_features, num_classes)
        elif "mnasnet" in model:
            net.classifier = nn.Sequential( nn.Dropout(0.2),nn.Linear(net.classifier[1].in_features, num_classes))
        else:
            net.fc = nn.Linear(net.fc.in_features, num_classes)

    # Move the model to the GPU if available
    net.to(device)

    # Specify loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0006, momentum=0.9)

    # Train the CNN
    num_epochs = 200
    best_epoch = 0
    best_accuracy = -1
    patience = 10


    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the data to the GPU
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}")
        f.write(f"{epoch + 1}\t{running_loss / (i + 1)}\t")

        # Evaluate the model on the test dataset
        net.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        correct_labels = []
        incorrect_labels = []

        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move the data to the GPU
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if epoch > 100:
                    correct_indices = (predicted == labels).nonzero(as_tuple=True)[0].tolist()
                    incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0].tolist()
                    for i in correct_indices:
                        correct_labels.append((predicted[i].item(), labels[i].item()))
                    for i in incorrect_indices:
                        incorrect_labels.append((predicted[i].item(), labels[i].item()))

                #correct += len(correct_indices)

                
        accuracy = 100 * correct / total
        print(f"Accuracy on the test dataset: {accuracy:.2f}%")
        f.write(f"{accuracy:.2f}\n")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(net, weights_file)
            
        if epoch - best_epoch >= patience:
           print(f"Stopping training at epoch {epoch} as accuracy did not improve for {patience} epochs.")
           break
        
        # if epoch > 100:
        #    print(f"Correctly predicted labels: {correct_labels}")
        #    print(f"Incorrectly predicted labels: {incorrect_labels}")
        print("\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"At epoch {best_epoch} reaches the highest accuracy {best_accuracy:.2f}%")
    f.write(f"\nAt epoch {best_epoch} reaches the highest accuracy {best_accuracy:.2f}%")
    f.close()
