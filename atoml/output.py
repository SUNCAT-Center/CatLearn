import time

localtime = time.asctime(time.localtime(time.time()))


def write_datasetup(function, data):
    """ Write output for the datasetup functions. """
    with open('ATOMLout.txt', 'a+') as writefile:
        if function is 'get_unique':
            writefile.write('Output from atoml.get_unique. Run on %s\n'
                            % localtime)
            writefile.write('index, target\n(training size is ' +
                            str(len(data['taken'])) + ')\n')
            for i, j in zip(data['taken'], data['target']):
                writefile.write(str(i) + ', ' + str(j) + '\n')
            writefile.write('\nEnd of get_unique function.\n \n')

        if function is 'get_train':
            writefile.write('Output from atoml.get_train. Run on %s\n'
                            % localtime)
            writefile.write('index, target\n(training size is ' +
                            str(len(data['order'])) + ')\n')
            for i, j in zip(data['order'], data['target']):
                writefile.write(str(i) + ', ' + str(j) + '\n')
            writefile.write('\nEnd of get_train function.\n \n')

        if function is 'data_split':
            writefile.write('Output from atoml.data_split. Run on %s\n'
                            % localtime)
            s = 'index set_1, target set_1, '
            for i in list(range(len(data['split_cand']) - 1)):
                s += 'index set_' + str(i + 2) + ', target set_' + str(i + 2)
            writefile.write(s + '\n(training size is ' +
                            str([len(data['split_cand'][i]) for i in
                                 list(range(len(data['split_cand'])))]) +
                            ')\n')
            for r in list(range(len(data['split_cand'][0]))):
                row = str(data['index'][0][r]) + ', ' + str(
                    data['target'][0][r])
                for l in list(range(len(data['split_cand']) - 1)):
                    while (len(data['index'][l + 1]) <
                           len(data['split_cand'][0])):
                        data['index'][l + 1].append(None)
                        data['target'][l + 1].append(None)
                    row += ', ' + str(data['index'][l + 1][r]) + ', ' + str(
                        data['target'][l + 1][r])
                writefile.write(str(row) + '\n')
            writefile.write('\nEnd of get_train function.\n \n')
