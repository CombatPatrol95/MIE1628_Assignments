# import multiprocessing
import os

# def map_function(chunk):
#     """Mapper function to count lines in a chunk of the file."""
#     return sum(1 for line in chunk) 

# def reduce_function(results):
#     """Reducer function to sum up the line counts from mappers."""
#     return sum(results)

# def process_chunk(args):
#     return map_function(open(args[0], 'r').readlines()[args[1]//100:args[2]//100])

# def count_lines_mapreduce(input_file, output_file, num_processes=multiprocessing.cpu_count()):
#     """Main function to orchestrate MapReduce."""
#     pool = multiprocessing.Pool(processes=num_processes) 

#     with open(input_file, 'r') as f:
#         file_size = f.seek(0, 2) 
#         chunk_size = file_size // num_processes 

#         chunks = [(input_file, i * chunk_size, (i + 1) * chunk_size if i < num_processes - 1 else file_size) 
#                   for i in range(num_processes)]

#         results = pool.map(process_chunk, chunks)

#         total_lines = reduce_function(results)

#     with open(output_file, 'w') as f:
#         f.write(str(total_lines))

# if __name__ == '__main__':
#     input_file = os.path.join(os.path.dirname(__file__), 'shakespeare.txt')
#     output_file = 'line_count_output.txt' 

#     count_lines_mapreduce(input_file, output_file)
from mrjob.job import MRJob

class LineCounter(MRJob):
    def mapper(self, _, line):
        yield None, 1  # Emit 1 for each line

    def reducer(self, _, counts):
        yield None, sum(counts)  # Sum the counts

if __name__ == '__main__':
    LineCounter.run()
