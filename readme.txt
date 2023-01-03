- All imported packages is written on 'import.txt' file.

- The file 'running_command.txt' contains command codes in terminal to help us 
  get the performance of the algorithm.


- Usage: 
      USAGE:      python pacman.py <options>
      EXAMPLES:   (1) python pacman.py
                      - starts an interactive game
                  (2) python pacman.py --layout smallClassic --zoom 2
                  OR  python pacman.py -l smallClassic -z 2
                      - starts an interactive game on a smaller board, zoomed in
      

    Options:
      -h, --help            show this help message and exit
      -n GAMES, --numGames=GAMES
                            the number of GAMES to play [Default: 1]
      -l LAYOUT_FILE, --layout=LAYOUT_FILE
                            the LAYOUT_FILE from which to load the map layout
                            [Default: mediumClassic]
      -p TYPE, --pacman=TYPE
                            the agent TYPE in the pacmanAgents module to use
                            [Default: KeyboardAgent]
      -t, --textGraphics    Display output as text only
      -q, --quietTextGraphics
                            Generate minimal output and no graphics
      -g TYPE, --ghosts=TYPE
                            the ghost agent TYPE in the ghostAgents module to use
                            [Default: RandomGhost]
      -k NUMGHOSTS, --numghosts=NUMGHOSTS
                            The maximum number of ghosts to use [Default: 4]
      -z ZOOM, --zoom=ZOOM  Zoom the size of the graphics window [Default: 1.0]
      -f, --fixRandomSeed   Fixes the random seed to always play the same game
      -r, --recordActions   Writes game histories to a file (named by the time
                            they were played)
      --replay=GAMETOREPLAY
                            A recorded game file (pickle) to replay
      -a AGENTARGS, --agentArgs=AGENTARGS
                            Comma separated values sent to agent. e.g.
                            "opt1=val1,opt2,opt3=val3"
      -x NUMTRAINING, --numTraining=NUMTRAINING
                            How many episodes are training (suppresses output)
                            [Default: 0]
      --frameTime=FRAMETIME
                            Time to delay between frames; <0 means keyboard
                            [Default: 0.1]
      -c, --catchExceptions
                            Turns on exception handling and timeouts during games
      --timeout=TIMEOUT     Maximum length of time an agent can spend computing in
                            a single game [Default: 30]
