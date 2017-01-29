import requests
import logging
import base64
import time
import os
from threading import Thread

corpus_dir = '/Users/ryanw_smith/PycharmProjects/mlb_solution/corpus'

class Server(object):
    url = 'https://mlb.praetorian.com'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.bin_b64 = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data=None):
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                if r.status_code == 500:
                    raise Exception('Unknown Server Exception')

                return r.json()
            except Exception as e:
                self.log.error(e)
                self.log.info('Waiting 60 seconds before next request')
                time.sleep(60)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])

        self.bin_b64 = r.get('binary', '')
        self.binary  = base64.b64decode(self.bin_b64)
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', self.hash)
        self.ans  = r.get('target', 'unknown')
        return r


def build_corpus(thread_num):
    import random

    of_path = os.path.join(corpus_dir, str(thread_num)+'_'+str(time.time())+'.txt')
    of = open(of_path, 'w')
    s = Server()

    for i in range(40000):
        # query the /challenge endpoint
        s.get()

        # choose a random target and /solve
        target = random.choice(s.targets)
        s.post(target)

        of.write(s.ans + ', ' + s.bin_b64+'\n')
        of.flush()

    of.close()


if __name__ == "__main__":
    thread_count = 25

    threads = []

    for i in range(thread_count):
        threads.append(Thread(target=build_corpus, args=(i,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
