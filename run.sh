#!/usr/bin/env bash

############################################~--|         <     >       |--~################################################
############################################~---|      <         >      |---~##############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#---------<-~-~-~o~-~-~->---------#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############################################~---|      <         >      |---~##############################################
############################################~--|         <     >       |--~################################################

usage() {
    echo "Usage: $0 [-m] <python_script_or_module>"
    echo "  -m: Run as a Python module (using python -m)"
    echo "  No flag: Run as a Python script"
    exit 1
}

MODE="script"

while getopts ":m" opt; do
  case $opt in
    m)
      MODE="module"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
  esac
done

shift $((OPTIND -1))

[ "$#" -ne 1 ] && usage

[ ! -f .env ] && echo ".env file not found" && exit 4

set -a
source .env
set +a

ARGUMENT="$1"

if [ "$MODE" = "module" ]; then
    time python -m "$ARGUMENT"
else
[ ! -f "$ARGUMENT" ] && echo "Python script '$ARGUMENT' not found" && exit 2
time python "$ARGUMENT"
fi
