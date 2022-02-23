var fs = require('fs');



var dirName = process.argv[2]


console.log(dirName)

var data = fs.readFileSync(dirName + '/' + "x_train_1_pre.csv","ascii").toString();

data = data.split('\n');


data = data.filter( line => {
                       return(line.substr(0,3) !== "NaN")
                   })


data = data.map( line => {
                    /*
                    if ( line.indexOf("totallyDone") > 0 ) {
                        line = line.replace("totallyDone",3);
                    } else if ( line.indexOf("finished")  > 0 ) {
                        line = line.replace("finished",1);
                    }

                    if ( line.indexOf("none") > 0 ) {
                        line = line.replace("none",0);
                    }
                    */
                    line = line.trim();
                    line = line.split(',');

                    /*
                    var date = line[7];

                    date = date.split(' ');
                    var ampm = date[1].trim();
                    var hourAdder = 0;
                    if ( ampm == "pm" ) {
                        hourAdder = 12;
                    }

                    var hrmin = date[0].split(':');
                    var hrs = parseInt(hrmin) + hourAdder;
                    var mins = parseInt(hrmin[1]);
                    line[7] = hrs*60 + mins;
                    */

                    line = [1].concat(line)

                    return(line);
                })


data = data.map( row => {
                    var y = row.splice(9,1);
                    var x = row;
                    return([x,y])
                });

//
var dataX = data.map(xy => {
                         return(xy[0])
                     })

//
var dataY = data.map(xy => {
                         return(parseInt(xy[1][0]))
                     })
//

var dataXTestKeys = {};

var n = dataX.length/3;
for ( var j = 0; j < n; j++ ) {
    //
    var k = Math.floor(Math.random()*dataX.length)
    while ( dataXTestKeys[k] !== undefined ) {
        k =  Math.floor(Math.random()*dataX.length)
    }
    //
    dataXTestKeys[k] = 1;
}


var dataXTestIndecies = Object.keys(dataXTestKeys);


var dataXTest = dataXTestIndecies.map( index => {
                                          return(dataX[index])
                                      })

dataXTestIndecies.sort(function(a, b){return b - a});


var n = dataXTestIndecies.length;
for ( var i = 0; i < n; i++ ) {
    var ii = dataXTestIndecies[i];
    dataX.splice(ii,1);
    dataY.splice(ii,1);
}


//
dataX = dataX.map(row => {
                      return(row.join(','))
                  })
dataX = dataX.join('\n')

//
dataY = dataY.join('\n')


//
dataXTest = dataXTest.map(row => {
                      return(row.join(','))
                  })

dataXTest = dataXTest.join('\n')


fs.writeFileSync(dirName + '/' + "testX_1.csv",dataXTest)
fs.writeFileSync(dirName + '/' + "trainX_1.csv",dataX)
fs.writeFileSync(dirName + '/' + "trainY_1.csv",dataY)


