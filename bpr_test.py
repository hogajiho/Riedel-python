# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nose.tools import *
from bpr_n import BPR
from numpy.random import randint

def test_bpr_train_stores_data(model_name):
    bpr = BPR(1, 2, 3, model_name)
    bpr.train([
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
    ], batch_size=4)
    assert_equal(bpr._train_users, set([0, 1]))
    assert_equal(bpr._train_items, set([0, 1, 2]))
    assert_equal(bpr._train_dict, {
        0: [ 1, 2 ],
        1: [ 0, 2 ],
    })

def test_bpr_train_and_test(model_name):
    bpr = BPR(10, 2000, 500, model_name)
    train_data = zip(randint(2000, size=10000), randint(500, size=10000))
    bpr.train(train_data, batch_size=50)
    print "train done"
    assert(bpr.test(train_data) > 0.8)
    print "test done"
    test_data = zip(randint(2000, size=10000), randint(500, size=10000))
    assert(bpr.test(test_data) > 0.4 and bpr.test(test_data) < 0.6)

def test_bpr_train_no_epochs():
    bpr = BPR(10, 100, 50)
    train_data = zip(randint(100, size=1000), randint(50, size=1000))
    bpr.train(train_data, epochs=0)
    assert(bpr.test(train_data) > 0.4 and bpr.test(train_data) < 0.6)

def test_bpr_predictions():
    bpr = BPR(10, 100, 50)
    train_data = zip(randint(100, size=1000), randint(50, size=1000))
    bpr.train(train_data, epochs=1)
    assert_equal(bpr.predictions(0).shape, (50,))
    assert_equal(bpr.prediction(0,0), bpr.predictions(0)[0])
    assert_equal(len(bpr.top_predictions(0, topn=20)), 20)

if __name__ == "__main__":
    #test_bpr_train_stores_data("n")
    #test_bpr_train_stores_data("nf")
    #test_bpr_train_and_test("f")
    #test_bpr_train_and_test("n")
    test_bpr_train_and_test("nf")



