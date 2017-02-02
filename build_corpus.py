"""
Usage: python build_corpus.py corpus_path thread_count

       corpus_path: full or relative path to existing folders where corpus files will be written [required]
       thread_count: number of concurrent threads to use to query mlb challenge server

Example: python build_corpus.py /path/to/corpus 20
"""
import logging
import time
import sys
import os
from threading import Thread

from MLB.server import Server


DEFAULT_THREAD_COUNT = 10

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def build_corpus(thread_num):
    import random
    log = logging.getLogger(__name__)

    of_path = os.path.join(corpus_dir, str(thread_num)+'_'+str(time.time())+'.txt')
    of = open(of_path, 'w')
    s = Server(log=log)

    log.info("Starting thread number: " + str(thread_num))
    for i in range(40000):
        try:
            # query the /challenge endpoint
            s.get()

            # choose a random target and /solve
            target = random.choice(s.targets)
            s.post(target)

            of.write(s.ans + ', ' + s.bin_b64+'\n')
            of.flush()

            if i%100 == 0:
                log.info("#{}: {}, {} ".format(i, s.ans, s.bin_b64))
        except Exception as e:
            log.info("Error getting corpus data: " + str(e))

    of.close()


if __name__ == "__main__":
    try:
        corpus_dir = sys.argv[1]
        if len(sys.argv) >= 3:
            thread_count = int(sys.argv[2])
        else:
            thread_count = DEFAULT_THREAD_COUNT
    except:
        print(__doc__)
        sys.exit(-1)

    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=build_corpus, args=(i,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
