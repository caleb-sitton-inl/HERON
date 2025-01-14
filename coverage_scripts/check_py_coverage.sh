#!/bin/bash
SCRIPT_DIRNAME=`dirname $(readlink -f "$0")`
HERON_DIR=`(cd $SCRIPT_DIRNAME/..; pwd)`
cd $HERON_DIR
RAVEN_DIR=`python -c 'from src._utils import get_raven_loc; print(get_raven_loc())'`

source $HERON_DIR/coverage_scripts/initialize_coverage.sh

# read command-line arguments
ARGS=()
for A in "$@"
do
  case $A in
    --re=*)
      export REGEX="${A#--re=}" # Removes "--re=" and puts regex value into env variable
      ;;
    *)
      ARGS+=("$A")
      ;;
  esac
done

if [[ "$REGEX" == "" ]] # No custom regex value
then
  export REGEX="HERON/tests" # Default regex value for run_tests
fi # else it's set to a custom string, so leave it

#coverage help run
SRC_DIR=`(cd src && pwd)`

export COVERAGE_RCFILE="$SRC_DIR/../coverage_scripts/.coveragerc"
SOURCE_DIRS=($SRC_DIR,$SRC_DIR/../templates/)
OMIT_FILES=($SRC_DIR/dispatch/twin_pyomo_test.py,$SRC_DIR/dispatch/twin_pyomo_test_rte.py,$SRC_DIR/dispatch/twin_pyomo_limited_ramp.py,$SRC_DIR/ArmaBypass.py)
EXTRA="--source=${SOURCE_DIRS[@]} --omit=${OMIT_FILES[@]} --parallel-mode "
export COVERAGE_FILE=`pwd`/.coverage

coverage erase
($RAVEN_DIR/run_tests "${ARGS[@]}" --re=$REGEX --python-command="coverage run $EXTRA ")
TESTS_SUCCESS=$?

## Prepare data and generate the html documents
coverage combine
coverage html

# See report_py_coverage.sh file for explanation of script separation
(bash $HERON_DIR/coverage_scripts/report_py_coverage.sh --data-file=$COVERAGE_FILE --coverage-rc-file=$COVERAGE_RCFILE)

if [ $TESTS_SUCCESS -ne 0 ]
then
  echo "run_tests finished but some tests failed"
fi

exit $TESTS_SUCCESS
