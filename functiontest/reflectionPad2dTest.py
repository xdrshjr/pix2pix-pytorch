import torch


if __name__ == '__main__':
    # 4*4输入向量
    input = torch.arange(16, dtype=torch.float).reshape(4, 4)
    # 填充前的input张量
    print('填充前的input张量')
    print(input)
    # 挤压
    input = input.unsqueeze(0)
    input = input.unsqueeze(0)

    # 以中间的张量为起点 向四周扩散填充 填充长度为3
    pad2 = torch.nn.ReflectionPad2d(3)
    res2 = pad2(input)

    print('填充后的input张量')
    print(input)
    print(pad2)
    print(res2[0][0][:])

    print('a')
