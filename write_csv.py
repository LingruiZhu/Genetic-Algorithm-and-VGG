import csv


def create_csv(file_path):
    # file_path = 'C:/Users/Zhu/PycharmProjects/genetic-algorithm/test'
    with open(file_path, 'w+', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["idx", "a", "b"]
        csv_write.writerow(["idx", "a", "b"])


def write_csv(file_path):
    with open(file_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        tup1 = [[1, 'relu', True]]
        data_row = [['1','2','3'], ['4', '5', '6']]
        csv_write.writerows(tup1)


file_path = 'C:/Users/Zhu/PycharmProjects/genetic-algorithm/test'
create_csv(file_path)
write_csv(file_path)
write_csv(file_path)