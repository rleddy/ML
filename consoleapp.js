
const fs = require('fs');

const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'game> '
});



/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


class TicTacToe {

    constructor() {

        this.board = [['-','-','-'],['-','-','-'],['-','-','-']]

    }


    print() {
        this.board.forEach(line => {
                               console.log(line.join('|'))
                           })
    }

    addToken(token,i,j) {
        if ( ( i >= 0 && i < 3 ) && ( j >= 0 && j < 3 ) ) {
            if ( this.board[i][j] === '-' ) {
                this.board[i][j] = token;
                return true
            }
        }
        return false
    }

    iamFull() {
        var f1 = this.board[0].join('').indexOf('-') >= 0
        var f2 = this.board[1].join('').indexOf('-') >= 0
        var f3 = this.board[2].join('').indexOf('-') >= 0
        return(!(f1 || f2 || f3))
    }

    aiMakeMove() {
        if ( !this.iamFull() ) {
            for ( var i = 0; i < 3; i++ ) {
                for ( var j = 0; j < 3; j++ ) {
                    if ( this.board[i][j] === '-' ) {
                        this.addToken('O',i,j);
                        return;
                    }
                }
            }
        }
        throw new Error("no moves!")
    }

    winner(token) {

        for ( var i = 0; i < 3; i++ ) {
            var win = (this.board[i][0] == token) && (this.board[i][1] == token) && (this.board[i][2] == token)
            if ( win ) return(true)
        }


        for ( var j = 0; j < 3; j++ ) {
            var win = (this.board[0][j] == token) && (this.board[1][j] == token) && (this.board[2][j] == token)
            if ( win ) return(true)
        }

        if ( (this.board[0][0] == token) &&  (this.board[1][1] == token) && (this.board[2][2] == token) ) return(true)
        if ( (this.board[0][2] == token) &&  (this.board[1][1] == token) && (this.board[2][0] == token) ) return(true)

        return(false)
    }


    computerWin() {
        return this.winner('O')
    }

    humanWin() {
        return this.winner('X')
    }
}





var gGame = new TicTacToe();



/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

var cmdList = ['exit', 'quit', 'time', 'move']

rl.prompt();


rl.on('line', (line) => {

          var dline = line.trim();
          var dataArray = dline.split(' ');

          var cmd = dataArray.shift().toLowerCase();

          switch ( cmd ) {

              case 'exit':
              case 'quit': {
                  process.exit(0);
                  break;
              }

              case 'time' : {
                  var tt = new Date();
                  console.log(tt.toString())
                  break;
              }

              case "move" : {

                  var i = dataArray[0]
                  var j = dataArray[1]

                  try {
                      if ( gGame.addToken('X',i,j) )  {
                          gGame.print()

                          gGame.aiMakeMove()
                          gGame.print()
                      } else {
                          console.log(`try again => ${i},${j} not empty`)
                      }
                  } catch ( e ) {
                      process.exit(0)
                  }


                  break;
              }

              default: {
                  console.log(`commands are: ${cmdList.join(', ')}`)
                  break;
              }

          }

          rl.prompt();

}).on('close', () => {
  console.log('Have a great day!');
  process.exit(0);
});

