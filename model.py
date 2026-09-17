from util_module import *

class Equivariant_Network(torch.nn.Module):

    def __init__(self, n_classes=256, sym_group = "Dihyderal", N = 4):

        super(Equivariant_Network, self).__init__()

        if sym_group == 'Dihyderal':
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)

        elif sym_group == 'Circular':
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        in_type = e2nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        self.input_type = in_type

        out_type = e2nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            e2nn.MaskModule(in_type, 64, margin=1),
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block1.out_type
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block2.out_type
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block3.out_type
        out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block4.out_type
        out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block5.out_type
        out_type = e2nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        self.gpool = e2nn.GroupPooling(out_type)

        c = self.gpool.out_type.size
        
#         print(c)

#         self.fully_net = nn.Sequential(nn.Dropout(0.5), nn.Linear(5184, 256))
        
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(5184, 256),
            torch.nn.BatchNorm1d(256),
            # torch.nn.ELU(inplace=True),
            # torch.nn.Linear(256, 256),
        )

    def forward(self, input: torch.Tensor):

        x = e2nn.GeometricTensor(input, self.input_type)
        # print("inside forward")
        # print(x.shape, x[0, 0, :, :], x[0, 0, :, :])
        # print(self.block1.state_dict()['1.weights'].shape, self.block1.state_dict()['1.weights'])
        x = self.block1(x)
        # print(x.shape, x[0, 0, :, :], x[0, 0, :, :])
        x = self.block2(x)
        # print(x.shape, x)
        x = self.pool1(x)
        # print(x.shape, x)

        x = self.block3(x)
        # print(x.shape, x)
        x = self.block4(x)
        # print(x.shape, x)
        x = self.pool2(x)
        # print(x.shape, x)

        x = self.block5(x)
        # print(x.shape, x)
        x = self.block6(x)
        # print(x.shape, x)

        x = self.pool3(x)
        # print(x.shape, x)

        x = self.gpool(x)
        # print(x.shape, x)
        x = x.tensor

        x = self.fully_net(x.reshape(x.shape[0], -1))
        # print(x.shape, x)
        # print("Final result of applying encoder")
        # print(x, x.cpu().detach().numpy().mean(), x.cpu().detach().numpy().std())
        return x

    
class MyResNet(torch.nn.Module):
    
    def __init__(self):
        super(MyResNet, self).__init__()
        self.resnet_model = models.resnet18(weights=None)
        self.resnet_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 256))
    
    def forward(self, x):
        return self.resnet_model(x)
    
  

class Classifier(torch.nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, feat):
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        # out = self.fc2(feat)
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input):
        out = self.layer(input)
        return out
    
encoder_dict = {"ENN": Equivariant_Network, "ResNet": MyResNet}

        