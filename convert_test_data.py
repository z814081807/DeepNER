import os
import json
from tqdm import trange


def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_test_data_to_json(test_dir, save_dir):

    test_examples = []


    # process test examples
    for i in trange(1500, 1997):
        with open(os.path.join(test_dir, f'{i}.txt'), encoding='utf-8') as f:
            text = f.read()

        test_examples.append({'id': i,
                              'text': text})

    save_info(save_dir, test_examples, 'test')


if __name__ == '__main__':
    test_dir = './tcdata/juesai'
    save_dir = './data/raw_data_random'
    convert_test_data_to_json(test_dir, save_dir)
    print('测试数据转换完成')

