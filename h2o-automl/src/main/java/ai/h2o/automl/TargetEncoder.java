package ai.h2o.automl;

import water.*;
import water.fvec.*;
import water.rapids.Rapids;
import water.rapids.Val;
import water.util.Log;
import water.util.TwoDimTable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static water.util.FrameUtils.getColumnIndexByName;

// TODO probably should call this logic from FrameUtils if we are going to expose TE for the Java API users.
public class TargetEncoder {

    public static class DataLeakageHandlingStrategy {
        public static final byte LeaveOneOut  =  0;
        public static final byte KFold  =  1;
        public static final byte None  =  2;
    }

    public static class BlendingParams extends Iced<BlendingParams> {
      private long k;
      private long f;

      BlendingParams(long k, long f) {
        this.k = k;
        this.f = f;
      }
    }

    /**
     * @param columnNamesToEncode names of columns to apply target encoding to
     * @param targetColumnName target column index
     * @param foldColumnName should contain index of column as String. TODO Change later into suitable type.
     */
    //TODO do we need to do this preparation before as a separate phase? because we are grouping twice.
    //TODO At least it seems that way in the case of KFold. But even if we need to preprocess for other types of TE calculations... we should not affect KFOLD case anyway.
    public Map<String, Frame> prepareEncodingMap(Frame data, String[] columnNamesToEncode, String targetColumnName, String foldColumnName) {

        // Validate input data. Not sure whether we should check some of these.
        // It will become clear when we decide if TE is going to be exposed to user or only integrated into AutoML's pipeline

        if(data == null) throw new IllegalStateException("Argument 'data' is missing, with no default");

        if(columnNamesToEncode == null || columnNamesToEncode.length == 0)
            throw new IllegalStateException("Argument 'columnsToEncode' is not defined or empty");

        if(targetColumnName == null || targetColumnName.equals(""))
            throw new IllegalStateException("Argument 'target' is missing, with no default");

        if(! checkAllTEColumnsAreCategorical(data, columnNamesToEncode))
            throw new IllegalStateException("Argument 'columnsToEncode' should contain only names of categorical columns");

        if(Arrays.asList(columnNamesToEncode).contains(targetColumnName)) {
            throw new IllegalArgumentException("Columns for target encoding contain target column.");
        }

        int targetIndex = getColumnIndexByName(data, targetColumnName);

        Frame  dataWithoutNAsForTarget = filterOutNAsFromTargetColumn(data, targetIndex);

        Frame dataWithEncodedTarget = ensureTargetColumnIsNumericOrBinaryCategorical(dataWithoutNAsForTarget, targetIndex);

        Map<String, Frame> columnToEncodingMap = new HashMap<String, Frame>();

        for ( String teColumnName: columnNamesToEncode) { // TODO maybe we can do it in parallel
            Frame teColumnFrame = null;
            int colIndex = getColumnIndexByName(dataWithEncodedTarget, teColumnName);
            String tree = null;
            if (foldColumnName == null) {
                tree = String.format("(GB %s [%d] sum %s \"all\" nrow %s \"all\")", dataWithEncodedTarget._key, colIndex, targetIndex, targetIndex);
            } else {
                int foldColumnIndex = getColumnIndexByName(dataWithEncodedTarget, foldColumnName);

                tree = String.format("(GB %s [%d, %d] sum %s \"all\" nrow %s \"all\")", dataWithEncodedTarget._key, colIndex, foldColumnIndex, targetIndex, targetIndex);
            }
            Val val = Rapids.exec(tree);
            teColumnFrame = val.getFrame();
            teColumnFrame._key = Key.make(dataWithEncodedTarget._key.toString() + "_" + teColumnName + "_encodingMap");
            DKV.put(teColumnFrame._key, teColumnFrame);

            renameColumn(teColumnFrame, "sum_"+ targetColumnName, "numerator");
            renameColumn(teColumnFrame, "nrow", "denominator");

            columnToEncodingMap.put(teColumnName, teColumnFrame);
        }

        dataWithEncodedTarget.delete();
        dataWithoutNAsForTarget.delete();

        return columnToEncodingMap;
    }

    Frame ensureTargetColumnIsNumericOrBinaryCategorical(Frame data, String targetColumnName) {
        return ensureTargetColumnIsNumericOrBinaryCategorical(data, getColumnIndexByName(data, targetColumnName));
    };

    Frame ensureTargetColumnIsNumericOrBinaryCategorical(Frame data, int targetIndex) {
        if (data.vec(targetIndex).isCategorical()){
            Vec targetVec = data.vec(targetIndex);
            if(targetVec.cardinality() == 2) {
                return transformBinaryTargetColumn(data, targetIndex);
            }
            else {
                throw new IllegalStateException("`target` must be a binary vector");
            }
        }
        else {
            if(! data.vec(targetIndex).isNumeric()) {
                throw new IllegalStateException("`target` must be a numeric or binary vector");
            }
            return data;
        }
    };

    Map<String, Frame> prepareEncodingMap(Frame data, int[] columnIndexesToEncode, int targetIndex) {
        String [] columnNamesToEncode = getColumnNamesBy(data, columnIndexesToEncode);
        return prepareEncodingMap(data, columnNamesToEncode, getColumnNameBy(data, targetIndex), null);
    }

    Map<String, Frame> prepareEncodingMap(Frame data, int[] columnIndexesToEncode, int targetIndex, String foldColumnName) {
        String [] columnNamesToEncode = getColumnNamesBy(data, columnIndexesToEncode);
        return prepareEncodingMap(data, columnNamesToEncode, getColumnNameBy(data, targetIndex), foldColumnName);
    }

    Map<String, Frame> prepareEncodingMap(Frame data, int[] columnIndexesToEncode, int targetIndex, int foldColumnIndex) {
        String [] columnNamesToEncode = getColumnNamesBy(data, columnIndexesToEncode);
        String foldColumnName = getColumnNameBy(data, foldColumnIndex);
        return prepareEncodingMap(data, columnNamesToEncode, getColumnNameBy(data, targetIndex), foldColumnName);
    }

    String[] getColumnNamesBy(Frame data, int[] columnIndexes) {
        String [] allColumnNames = data._names.clone();
        ArrayList<String> columnNames = new ArrayList<String>();

        for(int idx : columnIndexes) {
            columnNames.add(allColumnNames[idx]);
        }
        return columnNames.toArray(new String[columnIndexes.length]);
    }

    String getColumnNameBy(Frame data, int columnIndex) {
        String [] allColumnNames = data._names.clone();
        return allColumnNames[columnIndex];
    }

    Frame renameColumn(Frame fr, int indexOfColumnToRename, String newName) {
        String[] names = fr.names();
        names[indexOfColumnToRename] = newName;
        fr.setNames(names);
        return fr;
    }

    Frame renameColumn(Frame fr, String oldName, String newName) {
        return renameColumn(fr, getColumnIndexByName(fr, oldName), newName);
    }

    private Frame execRapidsAndGetFrame(String astTree) {
        Val val = Rapids.exec(astTree);
        Frame res = val.getFrame();
        res._key = Key.make();
        DKV.put(res);
        return res;
    }

    // We might want to introduce parameter that will change this behaviour. We can treat NA's as extra class.
    Frame filterOutNAsFromTargetColumn(Frame data, int targetColumnIndex) {
        return data.filterOutNAsInColumn(targetColumnIndex);
    }

    Frame transformBinaryTargetColumn(Frame data, int targetIndex)  {
        return data.asQuasiBinomial(targetIndex);
    }

    Frame getOutOfFoldData(Frame encodingMap, String foldColumnName, long currentFoldValue)  {
        int foldColumnIndexInEncodingMap = getColumnIndexByName(encodingMap, foldColumnName);
        return encodingMap.filterNotByValue(foldColumnIndexInEncodingMap, currentFoldValue);
    }

    long[] getUniqueValuesOfTheFoldColumn(Frame data, int columnIndex) {
        Vec uniqueValues = data.uniqueValuesBy(columnIndex).vec(0);
        long numberOfUniqueValues = uniqueValues.length();
        assert numberOfUniqueValues <= Integer.MAX_VALUE : "Number of unique values exceeded Integer.MAX_VALUE";

        int length = (int) numberOfUniqueValues; // We assume that fold column should not has that many different values and we will fit into node's memory.
        long[] uniqueValuesArr = new long[length];
        for(int i = 0; i < uniqueValues.length(); i++) {
            uniqueValuesArr[i] = uniqueValues.at8(i);
        }
        uniqueValues.remove();
        return uniqueValuesArr;
    }

    private boolean checkAllTEColumnsAreCategorical(Frame data, String[] columnsToEncode)  {
        for( String columnName : columnsToEncode) {
            int columnIndex = getColumnIndexByName(data, columnName);
            if(! data.vec(columnIndex).isCategorical()) return false;
        }
        return true;
    }

    Frame groupByTEColumnAndAggregate(Frame data, int teColumnIndex) {
        int numeratorColumnIndex = getColumnIndexByName(data, "numerator");
        int denominatorColumnIndex = getColumnIndexByName(data, "denominator");
        String astTree = String.format("(GB %s [%d] sum %d \"all\" sum %d \"all\")",
                data._key, teColumnIndex, numeratorColumnIndex, denominatorColumnIndex);
        return execRapidsAndGetFrame(astTree);
    }

    Frame rBind(Frame a, Frame b) {
        if(a == null) {
            assert b != null;
            return b;
        } else {
            String tree = String.format("(rbind %s %s)", a._key, b._key);
            return execRapidsAndGetFrame(tree);
        }
    }

    Frame mergeByTEColumnAndFold(Frame a, Frame holdoutEncodeMap, int teColumnIndexOriginal, int foldColumnIndexOriginal, int teColumnIndex ) {
        int foldColumnIndexInEncodingMap = getColumnIndexByName(holdoutEncodeMap, "foldValueForMerge");
        String astTree = String.format("(merge %s %s TRUE FALSE [%d, %d] [%d, %d] 'auto' )",
                a._key, holdoutEncodeMap._key, teColumnIndexOriginal, foldColumnIndexOriginal, teColumnIndex, foldColumnIndexInEncodingMap);
        return execRapidsAndGetFrame(astTree);
    }

    Frame mergeByTEColumn(Frame a, Frame b, int teColumnIndexOriginal, int teColumnIndex) {
        String astTree = String.format("(merge %s %s TRUE FALSE [%d] [%d] 'auto' )", a._key, b._key, teColumnIndexOriginal, teColumnIndex);
        return execRapidsAndGetFrame(astTree);
    }

    Frame imputeWithMean(Frame a, int columnIndex) {
        long numberOfNAs = a.vec(columnIndex).naCnt();
        if (numberOfNAs > 0) {
            String astTree = String.format("(h2o.impute %s %d 'mean' 'interpolate' [] _ _)", a._key, columnIndex);
            Rapids.exec(astTree);
            Log.warn(String.format("Frame with id = %s was imputed with mean ( %d rows were affected)", a._key, numberOfNAs));
        }
        return a;
    }

    Frame appendColumn(Frame a, long columnValue, String appendedColumnName ) {
        return a.addCon(appendedColumnName, columnValue);
    }

    double calculateGlobalMean(Frame fr) {
        Vec numeratorVec = fr.vec("numerator");
        Vec denominatorVec = fr.vec("denominator");
        return numeratorVec.mean() / denominatorVec.mean();
    }

    Frame calculateAndAppendBlendedTEEncoding(Frame fr, Frame encodingMap, String targetColumnName, String appendedColumnName) {
      int numeratorIndex = getColumnIndexByName(fr, "numerator");
      int denominatorIndex = getColumnIndexByName(fr, "denominator");

      double globalMeanForTargetClass = calculateGlobalMean(encodingMap);
      BlendingParams blendingParams = new BlendingParams(20, 10);

      Frame frameWithBlendedEncodings = new CalcEncodingsWithBlending(numeratorIndex, denominatorIndex, globalMeanForTargetClass, blendingParams).doAll(Vec.T_NUM, fr).outputFrame();
      fr.add(appendedColumnName, frameWithBlendedEncodings.anyVec());
      return fr;
    }

    static class CalcEncodingsWithBlending extends MRTask<CalcEncodingsWithBlending> {
      private double globalMean;
      private int numeratorIdx;
      private int denominatorIdx;
      private BlendingParams blendingParams;

      CalcEncodingsWithBlending(int numeratorIdx, int denominatorIdx, double globalMean, BlendingParams blendingParams) {
        this.numeratorIdx = numeratorIdx;
        this.denominatorIdx = denominatorIdx;
        this.globalMean = globalMean;
        this.blendingParams = blendingParams;
      }

      @Override
      public void map(Chunk cs[], NewChunk ncs[]) {
        Chunk num = cs[numeratorIdx];
        Chunk den = cs[denominatorIdx];
        NewChunk nc = ncs[0];
        for (int i = 0; i < num._len; i++) {
          if (num.isNA(i) || den.isNA(i))
            nc.addNA();
          else if (den.at8(i) == 0) {
            nc.addNum(globalMean);
          } else {
            double lambda = 1.0 / (1 + Math.exp((blendingParams.k - den.atd(i)) / blendingParams.f));
            double blendedValue = lambda * (num.atd(i) / den.atd(i)) + (1 - lambda) * globalMean;
            nc.addNum(blendedValue);
          }
        }
      }
    }

    Frame calculateAndAppendTEEncoding(Frame fr, Frame encodingMap, String targetColumnName, String appendedColumnName) {
      int numeratorIndex = getColumnIndexByName(fr, "numerator");
      int denominatorIndex = getColumnIndexByName(fr, "denominator");

      double globalMeanForTargetClass = calculateGlobalMean(encodingMap); // we can only operate on encodingsMap because `fr` could not have target column at all

      //I can do a trick with appending a zero column and then we can map there an encodings.
      Frame frameWithEncodings = new CalcEncodings(numeratorIndex, denominatorIndex, globalMeanForTargetClass).doAll(Vec.T_NUM, fr).outputFrame();
      fr.add(appendedColumnName, frameWithEncodings.anyVec()); // Can we just add(append)? Would the order be preserved?
      return fr;
    }


    static class CalcEncodings extends MRTask<CalcEncodings> {
      private double globalMean;
      private int numeratorIdx;
      private int denominatorIdx;

      CalcEncodings(int numeratorIdx, int denominatorIdx, double globalMean) {
        this.numeratorIdx = numeratorIdx;
        this.denominatorIdx = denominatorIdx;
        this.globalMean = globalMean;
      }

      @Override
      public void map(Chunk cs[], NewChunk ncs[]) {
        Chunk num = cs[numeratorIdx];
        Chunk den = cs[denominatorIdx];
        NewChunk nc = ncs[0];
        for (int i = 0; i < num._len; i++) {
          if (num.isNA(i) || den.isNA(i))
            nc.addNA();
          else if (den.at8(i) == 0) {
            nc.addNum(globalMean);
          } else {
            nc.addNum(num.atd(i) / den.atd(i));
          }
        }
      }
    }

    //TODO think about what value could be used for substitution ( maybe even taking into account target's value)
    private String getDenominatorIsZeroSubstitutionTerm(Frame fr, String targetColumnName, double globalMeanForTargetClass) {
      // This should happen only for Leave-One-Out case:
      // These groups have this singleness in common and we probably want to represent it somehow.
      // If we choose just global average then we just lose difference between single-row-groups that have different target values.
      // We can: 1) Group is so small that we even don't want to care about te_column's values.... just use Prior average.
      //         2) Count single-row-groups and calculate    #of_single_rows_with_target0 / #all_single_rows  ;  (and the same for target1)
      //TODO Introduce parameter for algorithm that will choose the way of calculating of the value that is being imputed.
      String denominatorIsZeroSubstitutionTerm;

      if(targetColumnName == null) { // When we calculating encodings for instances without target values.
        denominatorIsZeroSubstitutionTerm = String.format("%s", globalMeanForTargetClass);
      } else {
        int targetColumnIndex = getColumnIndexByName(fr, targetColumnName);
        double globalMeanForNonTargetClass = 1 - globalMeanForTargetClass;  // This is probably a bad idea to use frequencies for `0` class when we use frequencies for `1` class elsewhere
        denominatorIsZeroSubstitutionTerm = String.format("ifelse ( == (cols %s [%d]) 1) %f  %f", fr._key, targetColumnIndex, globalMeanForTargetClass, globalMeanForNonTargetClass);
      }
      return denominatorIsZeroSubstitutionTerm;
    }

    Frame addNoise(Frame fr, String applyToColumnName, double noiseLevel, double seed) {
        int appyToColumnIndex = getColumnIndexByName(fr, applyToColumnName);
        String tree = String.format("(:= %s (+ (cols %s [%d] ) (- (* (* (h2o.runif %s %f ) 2.0 ) %f ) %f ) ) [%d] [] )", fr._key, fr._key, appyToColumnIndex, fr._key, seed, noiseLevel, noiseLevel, appyToColumnIndex);
        return execRapidsAndGetFrame(tree);
    }

    Frame subtractTargetValueForLOO(Frame data, String targetColumnName) {
      int numeratorIndex = getColumnIndexByName(data, "numerator");
      int denominatorIndex = getColumnIndexByName(data, "denominator");
      int targetIndex = getColumnIndexByName(data, targetColumnName);

      new SubtractCurrentRowForLeaveOneOutTask(numeratorIndex, denominatorIndex, targetIndex).doAll(data);
      return data;
    }

    public static class SubtractCurrentRowForLeaveOneOutTask extends MRTask<SubtractCurrentRowForLeaveOneOutTask> {
      int numeratorIdx;
      int denominatorIdx;
      int targetIdx;

    public SubtractCurrentRowForLeaveOneOutTask(int numeratorIdx, int denominatorIdx, int targetIdx) {
      this.numeratorIdx = numeratorIdx;
      this.denominatorIdx = denominatorIdx;
      this.targetIdx = targetIdx;
    }

    @Override
    public void map(Chunk cs[]) {
      Chunk num = cs[numeratorIdx];
      Chunk den = cs[denominatorIdx];
      Chunk target = cs[targetIdx];
      for (int i = 0; i < num._len; i++) {
        if (! target.isNA(i)) {
          num.set(i, num.atd(i) - target.atd(i));
          den.set(i, den.atd(i) - 1);
        }
      }
    }
  }

    /**
     * Core method for applying pre-calculated encodings to the dataset. There are multiple overloaded methods that we will
     * probably be able to get rid off if we are not going to expose Java API for TE.
     * We can just stick to one signature that will suit internal representations  of the AutoML's pipeline.
     *
     * @param data dataset that will be used as a base for creation of encodings .
     * @param columnsToEncode set of columns names that we want to encode.
     * @param targetColumnName name of the column with respect to which we were computing encodings.
     * @param columnToEncodingMap map of the prepared encodings with the keys being the names of the columns.
     * @param dataLeakageHandlingStrategy see TargetEncoding.DataLeakageHandlingStrategy //TODO use common interface for stronger type safety.
     * @param foldColumnName numerical column that contains fold number the row is belong to.
     * @param withBlendedAvg whether to apply blending or not.
     * @param noiseLevel amount of noise to add to the final encodings.
     * @param seed we might want to specify particular values for reproducibility in tests.
     * @return
     */
    public Frame applyTargetEncoding(Frame data,
                                     String[] columnsToEncode,
                                     String targetColumnName,
                                     Map<String, Frame> columnToEncodingMap,
                                     byte dataLeakageHandlingStrategy,
                                     String foldColumnName,
                                     boolean withBlendedAvg,
                                     double noiseLevel,
                                     double seed,
                                     boolean isTrainOrValidSet) {

        if(noiseLevel < 0 )
            throw new IllegalStateException("`noiseLevel` must be non-negative");

        //TODO Should we remove string columns from `data` as it is done in R version (see: https://0xdata.atlassian.net/browse/PUBDEV-5266) ?

        Frame dataCopy = data.deepCopy(Key.make().toString());
        DKV.put(dataCopy);

        Frame dataWithAllEncodings = null ;
        if(isTrainOrValidSet) {
          Frame dataWithEncodedTarget = ensureTargetColumnIsNumericOrBinaryCategorical(dataCopy, targetColumnName);
          dataWithAllEncodings = dataWithEncodedTarget.deepCopy(Key.make().toString());
          DKV.put(dataWithAllEncodings);
          dataWithEncodedTarget.delete();
        }
        else {
          dataWithAllEncodings = dataCopy;
        }


        for ( String teColumnName: columnsToEncode) {

            String newEncodedColumnName = teColumnName + "_te";

            Frame dataWithMergedAggregations = null;
            Frame dataWithEncodings = null;
            Frame dataWithEncodingsAndNoise = null;

            Frame targetEncodingMap = columnToEncodingMap.get(teColumnName);

            int teColumnIndex = getColumnIndexByName(dataWithAllEncodings, teColumnName);
            Frame holdoutEncodeMap = null;

            switch (dataLeakageHandlingStrategy) {
                case DataLeakageHandlingStrategy.KFold:
                    assert isTrainOrValidSet : "Following calculations assume we can access target column but we can do this only on training and validation sets.";
                    if(foldColumnName == null)
                        throw new IllegalStateException("`foldColumn` must be provided for dataLeakageHandlingStrategy = KFold");

                    int teColumnIndexInEncodingMap = getColumnIndexByName(targetEncodingMap, teColumnName);

                    int foldColumnIndex = getColumnIndexByName(dataWithAllEncodings, foldColumnName);
                    long[] foldValues = getUniqueValuesOfTheFoldColumn(targetEncodingMap, 1);

                    Scope.enter();

                    // Following part is actually a preparation phase for KFold case. Maybe we should move it to prepareEncodingMap method.
                    try {
                        for (long foldValue : foldValues) {
                            Frame outOfFoldData = getOutOfFoldData(targetEncodingMap, foldColumnName, foldValue);

                            Frame groupedByTEColumnAndAggregate = groupByTEColumnAndAggregate(outOfFoldData, teColumnIndexInEncodingMap);

                            renameColumn(groupedByTEColumnAndAggregate, "sum_numerator", "numerator");
                            renameColumn(groupedByTEColumnAndAggregate, "sum_denominator", "denominator");

                            Frame groupedWithAppendedFoldColumn = groupedByTEColumnAndAggregate.addCon("foldValueForMerge", foldValue);

                            if (holdoutEncodeMap == null) {
                                holdoutEncodeMap = groupedWithAppendedFoldColumn;
                            } else {
                                Frame newHoldoutEncodeMap = rBind(holdoutEncodeMap, groupedWithAppendedFoldColumn);
                                holdoutEncodeMap.delete();
                                holdoutEncodeMap = newHoldoutEncodeMap;
                            }

                            outOfFoldData.delete();
                            Scope.track(groupedByTEColumnAndAggregate);
                        }
                    } finally {
                        Scope.exit();
                    }
                    // End of the preparation phase

                    dataWithMergedAggregations = mergeByTEColumnAndFold(dataWithAllEncodings, holdoutEncodeMap, teColumnIndex, foldColumnIndex, teColumnIndexInEncodingMap);

                    dataWithEncodings = calculateEncoding(dataWithMergedAggregations, targetEncodingMap, targetColumnName, newEncodedColumnName, withBlendedAvg);

                    dataWithEncodingsAndNoise = applyNoise(dataWithEncodings, newEncodedColumnName, noiseLevel, seed);

                    // if column is represented only in one fold then during computation of out-of-fold subsets we will get empty aggregations.
                    // When merging with the original dataset we will get NA'a on the right side
                    imputeWithMean(dataWithEncodingsAndNoise, getColumnIndexByName(dataWithEncodingsAndNoise, newEncodedColumnName));

                    removeNumeratorAndDenominatorColumns(dataWithEncodingsAndNoise);

                    dataWithAllEncodings.delete();
                    dataWithAllEncodings = dataWithEncodingsAndNoise.deepCopy(Key.make().toString());
                    DKV.put(dataWithAllEncodings);

                    dataWithEncodingsAndNoise.delete();
                    holdoutEncodeMap.delete();

                    break;
                case DataLeakageHandlingStrategy.LeaveOneOut:
                    assert isTrainOrValidSet : "Following calculations assume we can access target column but we can do this only on training and validation sets.";
                    foldColumnIsInEncodingMapCheck(foldColumnName, targetEncodingMap);

                    Frame groupedTargetEncodingMap = groupingIgnoringFordColumn(foldColumnName, targetEncodingMap, teColumnName);

                    int teColumnIndexInGroupedEncodingMap = getColumnIndexByName(groupedTargetEncodingMap, teColumnName);
                    dataWithMergedAggregations = mergeByTEColumn(dataWithAllEncodings, groupedTargetEncodingMap, teColumnIndex, teColumnIndexInGroupedEncodingMap);

                    Frame preparedFrame = subtractTargetValueForLOO(dataWithMergedAggregations,  targetColumnName);

                    dataWithEncodings = calculateEncoding(preparedFrame, groupedTargetEncodingMap, targetColumnName, newEncodedColumnName, withBlendedAvg); // do we really need to pass groupedTargetEncodingMap again?

                    dataWithEncodingsAndNoise = applyNoise(dataWithEncodings, newEncodedColumnName, noiseLevel, seed);

                    imputeWithMean(dataWithEncodingsAndNoise, getColumnIndexByName(dataWithEncodingsAndNoise, newEncodedColumnName));

                    removeNumeratorAndDenominatorColumns(dataWithEncodingsAndNoise);

                    dataWithAllEncodings.delete();
                    dataWithAllEncodings = dataWithEncodingsAndNoise.deepCopy(Key.make().toString());
                    DKV.put(dataWithAllEncodings);

                    preparedFrame.delete();
                    dataWithEncodingsAndNoise.delete();
                    groupedTargetEncodingMap.delete();

                    break;
                case DataLeakageHandlingStrategy.None:
                    foldColumnIsInEncodingMapCheck(foldColumnName, targetEncodingMap);
                    Frame groupedTargetEncodingMapForNone = groupingIgnoringFordColumn(foldColumnName, targetEncodingMap, teColumnName);
                    int teColumnIndexInGroupedEncodingMapNone = getColumnIndexByName(groupedTargetEncodingMapForNone, teColumnName);
                    dataWithMergedAggregations = mergeByTEColumn(dataWithAllEncodings, groupedTargetEncodingMapForNone, teColumnIndex, teColumnIndexInGroupedEncodingMapNone);

                    if(isTrainOrValidSet)
                      dataWithEncodings = calculateEncoding(dataWithMergedAggregations, groupedTargetEncodingMapForNone, targetColumnName, newEncodedColumnName, withBlendedAvg);
                    else
                      dataWithEncodings = calculateEncoding(dataWithMergedAggregations, groupedTargetEncodingMapForNone, null, newEncodedColumnName, withBlendedAvg);

                  // In cases when encoding has not seen some levels we will impute NAs with mean. Mean is a dataleakage btw.
                    // we'd better use stratified sampling for te_Holdout. Maybe even choose size of holdout taking into account size of the minimal set that represents all levels.
                    imputeWithMean(dataWithEncodings, getColumnIndexByName(dataWithEncodings, newEncodedColumnName));

                    dataWithEncodingsAndNoise = applyNoise(dataWithEncodings, newEncodedColumnName, noiseLevel, seed);

                    removeNumeratorAndDenominatorColumns(dataWithEncodingsAndNoise);

                    dataWithAllEncodings.delete();
                    dataWithAllEncodings = dataWithEncodingsAndNoise.deepCopy(Key.make().toString());
                    DKV.put(dataWithAllEncodings);

                    dataWithEncodingsAndNoise.delete();
                    groupedTargetEncodingMapForNone.delete();
            }

            dataWithMergedAggregations.delete();
            dataWithEncodings.delete();
        }

        dataCopy.delete();

        return dataWithAllEncodings;
    }

    private Frame calculateEncoding(Frame preparedFrame, Frame encodingMap, String targetColumnName, String newEncodedColumnName, boolean withBlendedAvg) {
        if (withBlendedAvg) {
            return calculateAndAppendBlendedTEEncoding(preparedFrame, encodingMap, targetColumnName, newEncodedColumnName);

        } else {
            return calculateAndAppendTEEncoding(preparedFrame, encodingMap, targetColumnName, newEncodedColumnName);
        }
    }

    private Frame applyNoise(Frame frameWithEncodings, String newEncodedColumnName, double noiseLevel, double seed) {
        if(noiseLevel > 0) {
            return addNoise(frameWithEncodings, newEncodedColumnName, noiseLevel, seed);
        } else {
            return frameWithEncodings;
        }
    }

    void removeNumeratorAndDenominatorColumns(Frame fr) {
        Vec removedNumeratorNone = fr.remove("numerator");
        removedNumeratorNone.remove();
        Vec removedDenominatorNone = fr.remove("denominator");
        removedDenominatorNone.remove();
    }

    private void foldColumnIsInEncodingMapCheck(String foldColumnName, Frame targetEncodingMap) {
        if(foldColumnName == null && targetEncodingMap.names().length > 3) {
            throw new IllegalStateException("Passed along encoding map possibly contains fold column. Please provide fold column name so that it becomes possible to regroup (by ignoring folds).");
        }
    }

    Frame groupingIgnoringFordColumn(String foldColumnName, Frame targetEncodingMap, String teColumnName) {
        if (foldColumnName != null) {
            int teColumnIndex = getColumnIndexByName(targetEncodingMap, teColumnName);

            Frame newTargetEncodingMap = groupByTEColumnAndAggregate(targetEncodingMap, teColumnIndex);
            renameColumn(newTargetEncodingMap, "sum_numerator", "numerator");
            renameColumn(newTargetEncodingMap, "sum_denominator", "denominator");
            return newTargetEncodingMap;
        } else {
            Frame targetEncodingMapCopy = targetEncodingMap.deepCopy(Key.make().toString());
            DKV.put(targetEncodingMapCopy);
            return targetEncodingMapCopy;
        }
    }

    public Frame applyTargetEncoding(Frame data,
                                     String[] columnsToEncode,
                                     String targetColumnName,
                                     Map<String, Frame> targetEncodingMap,
                                     byte dataLeakageHandlingStrategy,
                                     String foldColumn,
                                     boolean withBlendedAvg,
                                     double seed,
                                     boolean isTrainOrValidSet) {
        double defaultNoiseLevel = 0.01;
        double noiseLevel = 0.0;
        int targetIndex = getColumnIndexByName(data, targetColumnName);
        Vec targetVec = data.vec(targetIndex);
        if(targetVec.isNumeric()) {
            noiseLevel = defaultNoiseLevel * (targetVec.max() - targetVec.min());
        } else {
            noiseLevel = defaultNoiseLevel;
        }
        return this.applyTargetEncoding(data, columnsToEncode, targetColumnName, targetEncodingMap, dataLeakageHandlingStrategy, foldColumn, withBlendedAvg, noiseLevel, seed, isTrainOrValidSet);
    }

    public Frame applyTargetEncoding(Frame data,
                                     String[] columnsToEncode,
                                     String targetColumnName,
                                     Map<String, Frame> targetEncodingMap,
                                     byte dataLeakageHandlingStrategy,
                                     boolean withBlendedAvg,
                                     double seed,
                                     boolean isTrainOrValidSet) {
        return applyTargetEncoding(data, columnsToEncode, targetColumnName, targetEncodingMap, dataLeakageHandlingStrategy, null, withBlendedAvg, seed, isTrainOrValidSet);
    }

    public Frame applyTargetEncoding(Frame data,
                                     int[] columnIndexesToEncode,
                                     int targetIndex,
                                     Map<String, Frame> targetEncodingMap,
                                     byte dataLeakageHandlingStrategy,
                                     int foldColumnIndex,
                                     boolean withBlendedAvg,
                                     double seed,
                                     boolean isTrainOrValidSet) {
        String[] columnNamesToEncode = getColumnNamesBy(data, columnIndexesToEncode);
        String targetColumnName = getColumnNameBy(data, targetIndex);
        String foldColumnName = getColumnNameBy(data, foldColumnIndex);
        return this.applyTargetEncoding(data, columnNamesToEncode, targetColumnName, targetEncodingMap, dataLeakageHandlingStrategy, foldColumnName, withBlendedAvg, seed, isTrainOrValidSet);
    }

    public Frame applyTargetEncoding(Frame data,
                                     int[] columnIndexesToEncode,
                                     int targetIndex,
                                     Map<String, Frame> targetEncodingMap,
                                     byte dataLeakageHandlingStrategy,
                                     int foldColumnIndex,
                                     boolean withBlendedAvg,
                                     double noiseLevel,
                                     double seed,
                                     boolean isTrainOrValidSet) {
        String[] columnNamesToEncode = getColumnNamesBy(data, columnIndexesToEncode);
        String targetColumnName = getColumnNameBy(data, targetIndex);
        String foldColumnName = getColumnNameBy(data, foldColumnIndex);
        return this.applyTargetEncoding(data, columnNamesToEncode, targetColumnName, targetEncodingMap, dataLeakageHandlingStrategy, foldColumnName, withBlendedAvg, noiseLevel, seed, isTrainOrValidSet);
    }

    public Frame applyTargetEncoding(Frame data,
                                     int[] columnIndexesToEncode,
                                     int targetColumnIndex,
                                     Map<String, Frame> targetEncodingMap,
                                     byte dataLeakageHandlingStrategy,
                                     boolean withBlendedAvg,
                                     double noiseLevel,
                                     double seed,
                                     boolean isTrainOrValidSet) {
        String[] columnNamesToEncode = getColumnNamesBy(data, columnIndexesToEncode);
        String targetColumnName = getColumnNameBy(data, targetColumnIndex);
        return applyTargetEncoding(data, columnNamesToEncode, targetColumnName, targetEncodingMap, dataLeakageHandlingStrategy, withBlendedAvg, noiseLevel, seed, isTrainOrValidSet);
    }

    public Frame applyTargetEncoding(Frame data,
                                     String[] columnNamesToEncode,
                                     String targetColumnName,
                                     Map<String, Frame> targetEncodingMap,
                                     byte dataLeakageHandlingStrategy,
                                     boolean withBlendedAvg,
                                     double noiseLevel,
                                     double seed,
                                     boolean isTrainOrValidSet) {
        assert dataLeakageHandlingStrategy != DataLeakageHandlingStrategy.KFold : "Use another overloaded method for KFold dataLeakageHandlingStrategy.";
        return applyTargetEncoding(data, columnNamesToEncode, targetColumnName, targetEncodingMap, dataLeakageHandlingStrategy, null, withBlendedAvg, noiseLevel, seed, isTrainOrValidSet);
    }

    //TODO usefull during development remove
    public void checkNumRows(Frame before, Frame after) {
        long droppedCount = before.numRows()- after.numRows();
        if(droppedCount != 0) {
            Log.warn(String.format("Number of rows has dropped by %d after manipulations with frame ( %s , %s ).", droppedCount, before._key, after._key));
        }
    }

    // TODO usefull for development. remove.
    private void printOutFrameAsTable(Frame fr) {
        TwoDimTable twoDimTable = fr.toTwoDimTable();
        System.out.println(twoDimTable.toString());
    }

    // TODO usefull for development. remove.
    private void printOutFrameAsTable(Frame fr, boolean full, boolean rollups) {
        TwoDimTable twoDimTable = fr.toTwoDimTable(0, 1000000, rollups);
        System.out.println(twoDimTable.toString(2, full));
    }

    // TODO usefull for development. remove.
    private void printOutColumnsMeta(Frame fr) {
        for (String header : fr.toTwoDimTable().getColHeaders()) {
            String type = fr.vec(header).get_type_str();
            int cardinality = fr.vec(header).cardinality();
            System.out.println(header + " - " + type + String.format("; Cardinality = %d", cardinality));
        }
    }
}
