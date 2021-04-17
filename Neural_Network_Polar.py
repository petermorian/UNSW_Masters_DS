# spiral.py
# ZZEN9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.in_to_hid = nn.Linear(2,num_hid)     #two inputs
        self.hid_to_out = nn.Linear(num_hid,1)    #one output

    def forward(self, input):
        x = input[:,0]
        y = input[:,1]
        r = torch.sqrt(x*x + y*y).reshape(-1,1)         #polar transformation for r
        a = torch.atan2(y,x).reshape(-1,1)              #polar transformation for a
        polar_input = torch.cat((r, a),dim=1)           #concatenate r & a into single input
        self.hid1_sum = self.in_to_hid(polar_input)
        self.hidden1 = torch.tanh(self.hid1_sum)        #tanh activation
        out_sum = self.hid_to_out(self.hidden1)
        output = torch.sigmoid(out_sum)                 #sigmoid activation
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.in_to_hid1 = nn.Linear(2,num_hid)           #2 inputs (x,y)
        self.hid1_to_hid2 = nn.Linear(num_hid, num_hid) 
        self.hid2_to_out = nn.Linear(num_hid,1)          #1 output

    def forward(self, input):
        self.hid1_sum = self.in_to_hid1(input)
        self.hidden1 = torch.tanh(self.hid1_sum)        #tanh activation fully connected 1st layer
        self.hid2_sum = self.hid1_to_hid2(self.hidden1)
        self.hidden2 = torch.tanh(self.hid2_sum)        #tanh activation fully connected 2nd layer
        out_sum = self.hid2_to_out(self.hidden2)
        output = torch.sigmoid(out_sum)                 #sigmoid activation output 3rd layer
        return output

def graph_hidden(net, layer, node):
      # INSERT CODE HERE
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)
    with torch.no_grad(): 
        net.eval()    
        output = net(grid)   
        net.train()
        if layer == 1:
            pred = (net.hidden1[:, node]>=0).float()
        elif layer == 2:
            pred = (net.hidden2[:, node]>=0).float()
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')