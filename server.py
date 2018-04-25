import dlinputs as dli

data = dli.ittarreader("testdata/sample.tgz") | \
       dli.itmap(png=dli.pilrgb, cls=int)
dli.zmqserver(data, bind="tcp://*:17006")
