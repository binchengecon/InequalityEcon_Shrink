#! /bin/bash

git add -- . ':!job-outs' ':!bash'  ':!Figure' ':!SavedNets' ':!LargeData'
# git add -- . ':!job-outs' ':!bash' 
git commit -m 'new commit'
git push
