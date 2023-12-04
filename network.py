import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeStreamIQA(nn.Module):
    def __init__(self):
        super(ThreeStreamIQA, self).__init__()
        # LAB layer
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3)
       
        self.pool0 = nn.Conv2d(32, 32, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
      
        self.pool3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.convP1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        
        # self.pool4 = nn.Conv2d(256, 256, kernel_size=1, stride=2, padding=0)  # 1*1*256
        # gradient layer
        self.conv0_gra = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3) #32*32
       
        self.pool0_gra = nn.Conv2d(32, 32, kernel_size=7, stride=2, padding=3) #16*16
        self.conv1_gra = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)#16*16
        self.pool1_gra = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)#8*8
        self.conv2_gra = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) #88
        self.pool2_gra = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) #44
        self.conv3_gra = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)#44
        
        self.pool3_gra = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1) #22
        self.convP2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0) #22
     
        #self.pool4_gra = nn.Conv2d(256, 256, kernel_size=1, stride=2, padding=0)  # 1*1*256
        # MSCN layper
        self.conv0_MSCN = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
     
        self.pool0_MSCN = nn.Conv2d(32, 32, kernel_size=7, stride=2, padding=3)
        self.conv1_MSCN = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool1_MSCN = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_MSCN = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2_MSCN = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_MSCN = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4_MSCN = nn.BatchNorm2d(256)
        self.pool3_MSCN = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.convP3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
    
        # self.pool4_MSCN = nn.Conv2d(256, 256, kernel_size=1, stride=2, padding=0)  # 1*1*256
        # I2 layer

        # FC layer
        self.fc1 = nn.Conv2d(768, 800, kernel_size=1, stride=1, padding=0)

        self.fc2 = nn.Conv2d(800, 512, kernel_size=1, stride=1, padding=0)
        self.fc3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        # self.fc1 = nn.Linear(768, 800)
        # self.fc2 = nn.Linear(800, 512)
        # self.fc3 = nn.Linear(512, 1)

    def forward(self, input):
        x_LAB = input[0].view(-1, input[0].size(-3), input[0].size(-2), input[0].size(-1))
        x_gra = input[1].view(-1, input[1].size(-3), input[1].size(-2), input[1].size(-1))
        x_MSCN = input[2].view(-1, input[2].size(-3), input[2].size(-2), input[2].size(-1))
        # LAB region
        conv0 = F.relu(self.conv0(x_LAB))
        pool0 = self.pool0(conv0)

        conv1 = F.relu(self.conv1(pool0))
        pool1 = self.pool1(conv1)

        conv2 = F.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        conv3 = F.relu(self.conv3(pool2))
        pool3 = self.pool3(conv3)

        conv4 = F.relu(self.convP1(pool3))
   
        pool4 = F.adaptive_avg_pool2d(conv4, (1, 1))

        # gradient region
        conv0_gra = F.relu(self.conv0_gra(x_gra))
        pool0_gra = self.pool0_gra(conv0_gra)

        conv1_gra = F.relu(self.conv1_gra(pool0_gra))
        pool1_gra = self.pool1_gra(conv1_gra)

        conv2_gra = F.relu(self.conv2_gra(pool1_gra))
        pool2_gra = self.pool2_gra(conv2_gra)

        conv3_gra = F.relu(self.conv3_gra(pool2_gra))
        pool3_gra = self.pool3_gra(conv3_gra)

        conv4_gra = F.relu(self.convP2(pool3_gra))
   
        pool4_gra = F.adaptive_avg_pool2d(conv4_gra, (1, 1))
        # MSCN region
        conv0_MSCN = F.relu(self.conv0_MSCN(x_MSCN))
        pool0_MSCN = self.pool0_MSCN(conv0_MSCN)

        conv1_MSCN = F.relu(self.conv1_MSCN(pool0_MSCN))
        pool1_MSCN = self.pool1_MSCN(conv1_MSCN)

        conv2_MSCN = F.relu(self.conv2_MSCN(pool1_MSCN))
        pool2_MSCN = self.pool2_MSCN(conv2_MSCN)

        conv3_MSCN = F.relu(self.conv3_MSCN(pool2_MSCN))
        pool3_MSCN = self.pool3_MSCN(conv3_MSCN)

        conv4_MSCN = F.relu(self.convP2(pool3_MSCN))
      
        pool4_MSCN = F.adaptive_avg_pool2d(conv4_MSCN, (1, 1))

        # Feature squeeze
        #pool4 = pool4.squeeze(3)
        #pool4_gra = pool4_gra.squeeze(3)
        #pool4_MSCN = pool4_MSCN.squeeze(3)

        three_stream = torch.cat((pool4, pool4_gra, pool4_MSCN), 1)

        q = F.leaky_relu(self.fc1(three_stream), negative_slope=0.01)
        q = F.leaky_relu(self.fc2(q), negative_slope=0.01)
        q = F.leaky_relu(self.fc3(q), negative_slope=0.01)
      
        q = q.squeeze(3).squeeze(2)

        return q


