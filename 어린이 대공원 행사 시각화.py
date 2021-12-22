import matplotlib.pyplot as plt


def bar_plot(data):
    for key, values in data.items():
        plt.title(key + ' year, Count Events',fontsize=15)
        plt.bar(range(12), values, width=0.5)
        plt.xlabel('month', fontsize=15)
        plt.ylabel('count', fontsize=15)
        plt.xticks(range(12), range(1,13))
        plt.yticks(range(0,max(values)+5))
        plt.legend(['events'], fontsize=15)

        plt.tight_layout()
        plt.show()


def extract_info(txt_path):
    file = open(txt_path, 'r', encoding="UTF-8")
    contents = file.readlines()
    data = {'2018': [0 for _ in range(12)], '2019': [0 for _ in range(12)], '2020': [0 for _ in range(12)]}
    # total 38, 57, 2
    year = None

    for sentence in contents:
        if len(sentence) < 5:
            continue
        if len(sentence) < 10:
            is_year = sentence[1:-2]
            if is_year in data:
                year = is_year
            continue
        is_month = sentence[9:11]
        data[year][int(is_month) - 1] += 1

    return data


if __name__ == '__main__':
    txt_path = '행사월별.txt'
    data = extract_info(txt_path)
    bar_plot(data)