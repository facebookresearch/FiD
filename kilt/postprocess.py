import glob
import json

def write_kilt_format(glob_path, output_path):
    outfile = open(output_path, 'w')    
    results_path = glob.glob(glob_path)

    for path in results_path:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                d = {}
                try:
                    id, answer = line.split('\t')
                except ValueError:
                    print('error')
                id = int(id)
                answer = answer.split('\n')[0]
                #id = int(id)
                if id in d:
                    print('key already in dict', d[id], answer)
                d['id'] = id
                d['output'] = [{'answer':answer}]
                json.dump(d, outfile)
                outfile.write('\n')

#if __name__ == '__main__':
#    #path='/checkpoint/gizacard/qacache/test_/28023539/test_results/*'
#
#    glob_path='/checkpoint/gizacard/qacache/test/28026337/test_results/*'
#    output_path=open('nq_test_genere_kilt_results.jsonl', 'w')
#    write_kilt_format(glob_path, output_path)