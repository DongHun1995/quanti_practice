import torch
import torch.nn as nn

class NewModel(nn.module):
    def __init__(self, weight_path, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        #self.selected_out = OrderedDict()
        self.pretrained = Dongnet10()
        self.pretrained.load_state_dict(torch.load(weigth_pth))
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(1)))

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
    
    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out

model = NewModel(weigth_path, output_layers = [i for i in range(60)]).to('cuda')

#모델 훈련 돌리고
for data, target in valid_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    out, layerout = model(data)
    break


# 활성화 값 추출
count = 0
for k in layerout.keys():
    if k.startswith('leaky'):
        count += 1

quant_p = np.zeros((100, count+1))
batch = 100
for data, target in valid_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    out, layerout = model(data)
    col = 0
    for k in layerout.keys():
        if k.startswitch('leaky') or k=='conv10':
            if batch ==0:
                print(k)
            -, p = quantize(layerout[k].cpu.detach().numpy())
            quant_p[batch, col] = p
            col += 1
    batch += 1
    if batch == 100:
        break


# 최빈도값 계산
# find most frequent
def mode(X):
    # X : list
    return max(set(X), key=X.count)

# create looup table
cfg = []
for i in range(quant_p.shape[1]):
    print(i,mode(list(quant_p[:,i])))
    cfg.append(int(mode(list(quant_p[:,i]))))


#static quantization으로 모델 만들기
def quantize(self, X, p=0, NBIT=8):
    QRANGE = 2**(NBIT-1)
    p = torch.tensor(p, dtype=torch.int8).to('cuda')
    data_th = torch.clamp(X, -QRANGE*torch.float_power(2, -p), (QRANGE-1)*torch.float_power(2, -p))

    SCALE = torch.float_power(2, -p)

    data_qn = torch.round(data_th/SCALE)

    data_dqn = data_qn*SCALE
    return data_dqn

#그리고 forward에서 self.quantize(out, p=self.p_list[0]), 1, 2 이런식으로 접근