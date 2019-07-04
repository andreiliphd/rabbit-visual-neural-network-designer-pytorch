from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
import ast
from kivy.uix.label import Label
import time
# import matplotlib.pyplot as plt
# import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets

## In[2]:
#
#
# transforms_image = transforms.Compose([transforms.Resize(32),
#                                     transforms.CenterCrop(32),
#                                     transforms.ToTensor()])
# train_xray = torch.utils.data.DataLoader(datasets.ImageFolder('chest_xray/train',
#                                                                            transform=transforms_image),
#                                                        batch_size=20, shuffle=True)
# def imshow(img):
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """
## obtain one batch of training images
# dataiter = iter(train_xray)
# images, _ = dataiter.next() # _ for no labels
#
## plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(20, 4))
# plot_size=20
# for idx in np.arange(plot_size):
#    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
#    imshow(images[idx])
#

# In[3]:


# In[4]:


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

cnn = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=7, stride=2),
    nn.ReLU(inplace=True),
    Flatten(),
    nn.Linear(380192, 2),
)

model = cnn
class Settings():
    def __init__(self):
        self.data_folder = 'chest_xray/val'
        self.batch_size = 128
        self.shuffle = True
        self.cuda = torch.cuda.is_available()
        self.lr = 0.0005
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.num_of_epochs = 10

    def transform_image(self):
        return transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    def loader(self):
        return torch.utils.data.DataLoader(datasets.ImageFolder(self.data_folder,
                                                                transform=self.transform_image()),
                                           batch_size=self.batch_size, shuffle=self.shuffle)

    def train(self):
        output_string = str()
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        ## Define forward behavior
        for epoch in range(self.num_of_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(self.loader()):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                else:
                    data, target = data.cpu(), target.cpu()

                output = model(data)
                loss = self.loss(output, target)
                output_string = output_string + '\n' + 'Training loss: {:.6f}'.format(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            model.eval()
            output_string = output_string + '\n' + 'Epoch: ' + str(epoch)
            total_correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(self.loader()):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = self.loss(output, target)
                output_string = output_string + '\n' +  'Validation loss: {:.2%}'.format(loss.item())
                output_string = output_string + '\n' + 'Loss: ' + str(loss.item())
                max_arg_output = torch.argmax(output, dim=1)
                total_correct += int(torch.sum(max_arg_output == target))
                total += data.shape[0]
            output_string = output_string + '\n' + 'Validation accuracy: {:.0%}'.format(total_correct / total)
        return 'Validation accuracy: {:.0%}'.format(total_correct / total)
            # if total_correct / total > 0.8:
            #     torch.save(model.state_dict(), 'pt/XRP_' + str(time.strftime("%Y%m%d_%H%M%S")) + '.pt')


settings = Settings()

class MessagePopup(Popup):
    def __init__(self, title, message):
        super(MessagePopup, self).__init__(title=title)
        self.ids.message.text = message


class ConvPopup(Popup):
    def __init__(self, instance):
        super(ConvPopup, self).__init__(title='Convolutional layers settings')
        self.instace = instance
    def on_release(self):
        index = len(cnn._modules)
        try:
            cnn._modules[str(index)] = torch.nn.Conv2d(in_channels=ast.literal_eval(self.ids.in_channels.text),
                                                       out_channels=ast.literal_eval(self.ids.out_channels.text),
                                                       kernel_size=ast.literal_eval(self.ids.kernel.text),
                                                       stride=ast.literal_eval(self.ids.stride.text),
                                                       padding=ast.literal_eval(self.ids.padding.text)

                                                       )

        except Exception as e:
            popup = MessagePopup('Error', str(e))
            popup.open()
            return False

        self.dismiss()
        self.instace.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]

class MaxPoolPopup(Popup):
    def __init__(self, instance):
        super(MaxPoolPopup, self).__init__(title='MaxPool layers settings')
        self.instace = instance
    def on_release(self):
        index = len(cnn._modules)
        try:
            cnn._modules[str(index)] = torch.nn.MaxPool2d(kernel_size=ast.literal_eval(self.ids.kernel.text),
                                                          stride=ast.literal_eval(self.ids.stride.text),
                                                          padding=ast.literal_eval(self.ids.padding.text)
                                                       )

        except Exception as e:
            popup = MessagePopup('Error', str(e))
            popup.open()
            return False

        self.dismiss()
        self.instace.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]

class LinearPopup(Popup):
    def __init__(self, instance):
        super(LinearPopup, self).__init__(title='Linear layers settings')
        self.instace = instance
    def on_release(self):
        index = len(cnn._modules)
        try:
            cnn._modules[str(index)] = torch.nn.Linear(in_features=ast.literal_eval(self.ids.in_features.text),
                                                       out_features=ast.literal_eval(self.ids.out_features.text),
                                                       bias=ast.literal_eval(self.ids.bias.text)
                                                       )

        except Exception as e:
            popup = MessagePopup('Error', str(e))
            popup.open()
            return False

        self.dismiss()
        self.instace.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]

class LayerDesigner(BoxLayout):
    def __init__(self, **kwargs):
        super(LayerDesigner, self).__init__(**kwargs)
        self.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]
        self.ids.shuffle.active = settings.shuffle
        self.ids.cuda.active = settings.cuda
        self.ids.data_dir.text = settings.data_folder

    def on_press(self):
        print(self)

    def add_conv(self):
        popup = ConvPopup(self)
        popup.open()

    def add_maxpool(self):
        popup = MaxPoolPopup(self)
        popup.open()

    def add_linear(self):
        popup = LinearPopup(self)
        popup.open()

    def add_relu(self):
        index = len(cnn._modules)
        cnn._modules[str(index)] = torch.nn.ReLU(),
        self.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]

    def add_sigmoid(self):
        index = len(cnn._modules)
        cnn._modules[str(index)] = torch.nn.Sigmoid(),
        self.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]

    def add_tanh(self):
        index = len(cnn._modules)
        cnn._modules[str(index)] = torch.nn.Tanh(),
        self.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]

    def add_flatten(self):
        index = len(cnn._modules)
        cnn._modules[str(index)] = Flatten(),
        self.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]

    def remove_last_layer(self):
        print(cnn)
        index = len(cnn._modules)
        try:
            cnn._modules.pop(str(index - 1))
        except Exception:
            popup = MessagePopup('Error', 'There is nothing to remove!')
            popup.open()
            return False

        self.ids.rv.data = [{'text': str(x)} for x in cnn._modules.values()]
        print(cnn)

    def train(self):
        num_epochs = 1
        try:
            num_epochs = int(self.ids.num_epochs.text)
            settings.num_of_epochs = num_epochs
            lr = float(self.ids.lr.text)
            settings.lr = lr
            batch_size = int(self.ids.batch_size.text)
            settings.batch_size = batch_size
            shuffle = int(self.ids.shuffle.active)
            settings.shuffle = shuffle

        except ValueError as e:
            popup = MessagePopup('Error', str(e))
            popup.open()
            return False
        try:
            self.ids.console.text = settings.train()
        except Exception as e:
            popup = MessagePopup('Error', str(e))
            popup.open()
            return False


class LayerDesignerApp(App):
    def get_application_name(self):
        return "Recycleview sample -unselectable-"

    def build(self):
        return LayerDesigner()


if __name__ == '__main__':
    LayerDesignerApp().run()
