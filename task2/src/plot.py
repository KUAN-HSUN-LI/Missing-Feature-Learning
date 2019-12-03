import matplotlib.pyplot as plt
import json
import ipdb
# ipdb.set_trace()

plt.title('Acc')

# with open('baseline2/test_vanilla/history.json', 'r') as f:
#     h = json.loads(f.read())
# train = [l['loss'] for l in h['train']]
# valid = [l['loss'] for l in h['valid']]
# train = [l['accuracy'] for l in h['train']]
# valid = [l['accuracy'] for l in h['valid']]
# plt.plot(train, label='vanilla train')
# plt.plot(valid, label='vanilla valid')

# with open('baseline2/test_BN/history.json', 'r') as f:
#     h = json.loads(f.read())
# train = [l['loss'] for l in h['train']]
# valid = [l['loss'] for l in h['valid']]
# train = [l['accuracy'] for l in h['train']]
# valid = [l['accuracy'] for l in h['valid']]
# plt.plot(train, '--', label='BN train')
# plt.plot(valid, '--', label='BN valid')

with open('baseline1/baseline/history.json', 'r') as f:
    h = json.loads(f.read())
train = [l['accuracy'] for l in h['train']]
valid = [l['accuracy'] for l in h['valid']]
plt.plot(train, label='Orig train')
# plt.plot(valid, label='Orig valid')

with open('baseline1/BCEWL/history.json', 'r') as f:
    h = json.loads(f.read())
train = [l['accuracy'] for l in h['train']]
valid = [l['accuracy'] for l in h['valid']]
plt.plot(train, label='Logit train')
# plt.plot(valid, label='Logit valid')

# with open('baseline2/test_DP/history.json', 'r') as f:
#     h = json.loads(f.read())
# train_loss = [l['loss'] for l in h['train']]
# valid_loss = [l['loss'] for l in h['valid']]
# train_acc = [l['accuracy'] for l in h['train']]
# valid_acc = [l['accuracy'] for l in h['valid']]
# plt.plot(train_acc, ':', label='DP train')
# plt.plot(valid_acc, ':', label='DP valid')

# with open('baseline2/test_BN_DP/history.json', 'r') as f:
#     h = json.loads(f.read())
# train_loss = [l['loss'] for l in h['train']]
# valid_loss = [l['loss'] for l in h['valid']]
# train_acc = [l['accuracy'] for l in h['train']]
# valid_acc = [l['accuracy'] for l in h['valid']]
# plt.plot(train_acc, ':', label='BN_DP train')
# plt.plot(valid_acc, ':', label='BN_DP valid')


plt.legend(loc='best')
plt.savefig('plot')
