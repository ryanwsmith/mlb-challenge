import logging
import requests
import base64
import time


class Server(object):
    url = 'https://mlb.praetorian.com'

    def __init__(self, log=None):
        self.session = requests.session()
        self.binary = None
        self.bin_b64 = None
        self.hash = None
        self.wins = 0
        self.targets = []
        self.accuracy = 0.0

        if log is not None:
            self.log = log
        else:
            self.log = logging.getLogger(__name__)

    def _request(self, route, method='get', data=None):
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                elif r.status_code == 500:
                    raise Exception('Unknown Server Exception')
                elif r.status_code != 200:
                    self.log.info('Status Code: ' + str(r.status_code))

                return r.json()
            except Exception as e:
                self.log.error(e)

                # go home, your drunk (sleep off the aggression)
                if r.status_code == 429:  # Rate Limit Exception
                    self.log.info('Waiting 60 seconds before next request')
                    time.sleep(60)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])
        self.bin_b64 = r.get('binary', '')
        self.binary = base64.b64decode(self.bin_b64)
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.log.debug("RESPONSE: " + str(r))
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', self.hash)
        self.ans = r.get('target', 'unknown')
        self.accuracy = r.get('accuracy', -1)
        return r
