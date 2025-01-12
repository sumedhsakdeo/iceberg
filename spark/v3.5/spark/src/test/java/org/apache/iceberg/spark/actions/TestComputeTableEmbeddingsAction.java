/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.spark.actions;

import static org.apache.iceberg.spark.actions.TextEmbeddingUtil.EMBEDDINGS_V1_BLOB_PROPERTY;
import static org.apache.iceberg.types.Types.NestedField.optional;
import static org.apache.iceberg.types.Types.NestedField.required;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNoException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.apache.iceberg.BlobMetadata;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.Files;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Schema;
import org.apache.iceberg.StatisticsFile;
import org.apache.iceberg.Table;
import org.apache.iceberg.actions.ComputeTableEmbeddings;
import org.apache.iceberg.data.FileHelpers;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableMap;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.spark.CatalogTestBase;
import org.apache.iceberg.spark.Spark3Util;
import org.apache.iceberg.spark.SparkSchemaUtil;
import org.apache.iceberg.spark.SparkWriteOptions;
import org.apache.iceberg.spark.data.RandomData;
import org.apache.iceberg.spark.source.SimpleRecord;
import org.apache.iceberg.types.Types;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.parser.ParseException;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.TestTemplate;

public class TestComputeTableEmbeddingsAction extends CatalogTestBase {

  // In-process model
  private static final String MODEL_NAME = EmbeddingModelBuilder.ALL_MINI_LM_L6V2;
  private static final Map<String, String> MODEL_INPUTS = Collections.emptyMap();

  private static final Types.StructType LEAF_STRUCT_TYPE =
      Types.StructType.of(
          optional(1, "leafLongCol", Types.LongType.get()),
          optional(2, "leafStringCol", Types.StringType.get()));

  private static final Types.StructType NESTED_STRUCT_TYPE =
      Types.StructType.of(required(3, "leafStructCol", LEAF_STRUCT_TYPE));

  private static final Schema NESTED_SCHEMA =
      new Schema(required(4, "nestedStructCol", NESTED_STRUCT_TYPE));

  private static final Schema SCHEMA_WITH_NESTED_COLUMN =
      new Schema(
          required(4, "nestedStructCol", NESTED_STRUCT_TYPE),
          required(5, "stringCol", Types.StringType.get()));

  @TestTemplate
  public void testLoadingTableDirectly() {
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    sql("INSERT into %s values(1, 'abcd')", tableName);

    Table table = validationCatalog.loadTable(tableIdent);

    SparkActions actions = SparkActions.get();
    ComputeTableEmbeddings.Result results =
        actions
            .computeTableEmbeddings(table)
            .modelName(MODEL_NAME)
            .modelInputs(MODEL_INPUTS)
            .columns("data")
            .execute();
    StatisticsFile statisticsFile = results.statisticsFile();
    assertThat(statisticsFile.fileSizeInBytes()).isGreaterThan(0);
    assertThat(statisticsFile.blobMetadata()).hasSize(1);
  }

  @TestTemplate
  public void testComputeTableEmbeddingsAction() throws NoSuchTableException, ParseException {
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    Table table = Spark3Util.loadIcebergTable(spark, tableName);

    // To create multiple splits on the mapper
    table
        .updateProperties()
        .set("read.split.target-size", "100")
        .set("write.parquet.row-group-size-bytes", "100")
        .commit();
    List<SimpleRecord> records =
        Lists.newArrayList(
            new SimpleRecord(1, "a"),
            new SimpleRecord(1, "a"),
            new SimpleRecord(2, "b"),
            new SimpleRecord(3, "c"),
            new SimpleRecord(4, "d"));
    spark.createDataset(records, Encoders.bean(SimpleRecord.class)).writeTo(tableName).append();
    SparkActions actions = SparkActions.get();
    table.refresh();
    ComputeTableEmbeddings.Result results =
        actions
            .computeTableEmbeddings(table)
            .modelName(MODEL_NAME)
            .modelInputs(MODEL_INPUTS)
            .columns("data")
            .execute();
    assertThat(results).isNotNull();

    List<StatisticsFile> statisticsFiles = table.statisticsFiles();
    assertThat(statisticsFiles).hasSize(1);

    StatisticsFile statisticsFile = statisticsFiles.get(0);
    assertThat(statisticsFile.fileSizeInBytes()).isGreaterThan(0);
    assertThat(statisticsFile.blobMetadata()).hasSize(1);

    BlobMetadata blobMetadata = statisticsFile.blobMetadata().get(0);
    assertThat(blobMetadata.properties()).containsEntry(EMBEDDINGS_V1_BLOB_PROPERTY, "5");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsForInvalidColumns()
      throws NoSuchTableException, ParseException {
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    // Append data to create snapshot
    sql("INSERT into %s values(1, 'abcd')", tableName);
    Table table = Spark3Util.loadIcebergTable(spark, tableName);
    SparkActions actions = SparkActions.get();
    assertThatThrownBy(() -> actions.computeTableEmbeddings(table).columns("id1").execute())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageStartingWith("Can't find column id1 in table");
    assertThatThrownBy(() -> actions.computeTableEmbeddings(table).columns("id").execute())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageStartingWith("Can't compute embeddings on non-string type columns: id (int)");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsWithNoSnapshots()
      throws NoSuchTableException, ParseException {
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    Table table = Spark3Util.loadIcebergTable(spark, tableName);
    SparkActions actions = SparkActions.get();
    ComputeTableEmbeddings.Result result =
        actions.computeTableEmbeddings(table).columns("id").execute();
    assertThat(result.statisticsFile()).isNull();
  }

  @TestTemplate
  public void testComputeTableEmbeddingsWithNullValues()
      throws NoSuchTableException, ParseException {
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    List<SimpleRecord> records =
        Lists.newArrayList(
            new SimpleRecord(1, null),
            new SimpleRecord(1, "a"),
            new SimpleRecord(2, "b"),
            new SimpleRecord(3, "c"),
            new SimpleRecord(4, "d"));
    spark
        .createDataset(records, Encoders.bean(SimpleRecord.class))
        .coalesce(1)
        .writeTo(tableName)
        .append();
    Table table = Spark3Util.loadIcebergTable(spark, tableName);
    SparkActions actions = SparkActions.get();
    ComputeTableEmbeddings.Result results =
        actions
            .computeTableEmbeddings(table)
            .columns("data")
            .modelName(MODEL_NAME)
            .modelInputs(MODEL_INPUTS)
            .execute();
    assertThat(results).isNotNull();

    List<StatisticsFile> statisticsFiles = table.statisticsFiles();
    assertThat(statisticsFiles).hasSize(1);

    StatisticsFile statisticsFile = statisticsFiles.get(0);
    assertThat(statisticsFile.fileSizeInBytes()).isGreaterThan(0);
    assertThat(statisticsFile.blobMetadata()).hasSize(1);

    assertThat(statisticsFile.blobMetadata().get(0).properties())
        .containsEntry(EMBEDDINGS_V1_BLOB_PROPERTY, "4");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsWithSnapshotHavingDifferentSchemas()
      throws NoSuchTableException, ParseException {
    SparkActions actions = SparkActions.get();
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    // Append data to create snapshot
    sql("INSERT into %s values(1, 'abcd')", tableName);
    long snapshotId1 = Spark3Util.loadIcebergTable(spark, tableName).currentSnapshot().snapshotId();
    // Snapshot id not specified
    Table table = Spark3Util.loadIcebergTable(spark, tableName);

    assertThatNoException()
        .isThrownBy(
            () ->
                actions
                    .computeTableEmbeddings(table)
                    .modelName(MODEL_NAME)
                    .modelInputs(MODEL_INPUTS)
                    .columns("data")
                    .execute());

    sql("ALTER TABLE %s DROP COLUMN %s", tableName, "data");
    // Append data to create snapshot
    sql("INSERT into %s values(1)", tableName);
    table.refresh();
    long snapshotId2 = Spark3Util.loadIcebergTable(spark, tableName).currentSnapshot().snapshotId();

    // Snapshot id specified
    assertThatNoException()
        .isThrownBy(
            () ->
                actions
                    .computeTableEmbeddings(table)
                    .snapshot(snapshotId1)
                    .modelName(MODEL_NAME)
                    .modelInputs(MODEL_INPUTS)
                    .columns("data")
                    .execute());

    assertThatThrownBy(
            () ->
                actions
                    .computeTableEmbeddings(table)
                    .snapshot(snapshotId2)
                    .modelName(MODEL_NAME)
                    .modelInputs(MODEL_INPUTS)
                    .columns("data")
                    .execute())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageStartingWith("Can't find column data in table");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsWhenSnapshotIdNotSpecified()
      throws NoSuchTableException, ParseException {
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    // Append data to create snapshot
    sql("INSERT into %s values(1, 'abcd')", tableName);
    Table table = Spark3Util.loadIcebergTable(spark, tableName);
    SparkActions actions = SparkActions.get();
    ComputeTableEmbeddings.Result results =
        actions
            .computeTableEmbeddings(table)
            .modelName(MODEL_NAME)
            .modelInputs(MODEL_INPUTS)
            .columns("data")
            .execute();

    assertThat(results).isNotNull();

    List<StatisticsFile> statisticsFiles = table.statisticsFiles();
    assertThat(statisticsFiles).hasSize(1);

    StatisticsFile statisticsFile = statisticsFiles.get(0);
    assertThat(statisticsFile.fileSizeInBytes()).isGreaterThan(0);
    assertThat(statisticsFile.blobMetadata()).hasSize(1);

    assertThat(statisticsFile.blobMetadata().get(0).properties())
        .containsEntry(EMBEDDINGS_V1_BLOB_PROPERTY, "1");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsWithNestedSchema()
      throws NoSuchTableException, ParseException, IOException {
    List<Record> records = Lists.newArrayList(createNestedRecord());
    Table table =
        validationCatalog.createTable(
            tableIdent,
            SCHEMA_WITH_NESTED_COLUMN,
            PartitionSpec.unpartitioned(),
            ImmutableMap.of());
    DataFile dataFile = FileHelpers.writeDataFile(table, Files.localOutput(temp.toFile()), records);
    table.newAppend().appendFile(dataFile).commit();

    Table tbl = Spark3Util.loadIcebergTable(spark, tableName);
    SparkActions actions = SparkActions.get();
    actions
        .computeTableEmbeddings(tbl)
        .modelName(MODEL_NAME)
        .modelInputs(MODEL_INPUTS)
        .columns("nestedStructCol.leafStructCol.leafStringCol")
        .execute();

    tbl.refresh();
    List<StatisticsFile> statisticsFiles = tbl.statisticsFiles();
    assertThat(statisticsFiles).hasSize(1);
    StatisticsFile statisticsFile = statisticsFiles.get(0);
    assertThat(statisticsFile.fileSizeInBytes()).isGreaterThan(0);
    assertThat(statisticsFile.blobMetadata()).hasSize(1);
  }

  @TestTemplate
  public void testComputeTableEmbeddingWithNoComputableColumns() throws IOException {
    List<Record> records = Lists.newArrayList(createNestedRecord());
    Table table =
        validationCatalog.createTable(
            tableIdent, NESTED_SCHEMA, PartitionSpec.unpartitioned(), ImmutableMap.of());
    DataFile dataFile = FileHelpers.writeDataFile(table, Files.localOutput(temp.toFile()), records);
    table.newAppend().appendFile(dataFile).commit();

    table.refresh();
    SparkActions actions = SparkActions.get();
    assertThatThrownBy(() -> actions.computeTableEmbeddings(table).execute())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessage("No columns found to compute embeddings");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnByteColumn() throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("byte_col", "TINYINT");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnShortColumn()
      throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("short_col", "SMALLINT");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnIntColumn() throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("int_col", "INT");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnLongColumn() throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("long_col", "BIGINT");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnTimestampColumn()
      throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("timestamp_col", "TIMESTAMP");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnTimestampNtzColumn()
      throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("timestamp_col", "TIMESTAMP_NTZ");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnDateColumn() throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("date_col", "DATE");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnDecimalColumn()
      throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("decimal_col", "DECIMAL(20, 2)");
  }

  @TestTemplate
  public void testComputeTableEmbeddingsOnBinaryColumn()
      throws NoSuchTableException, ParseException {
    testComputeTableEmbeddingsAllInvalidColumns("binary_col", "BINARY");
  }

  public void testComputeTableEmbeddingsAllInvalidColumns(String columnName, String type)
      throws NoSuchTableException, ParseException {
    sql("CREATE TABLE %s (id int, %s %s) USING iceberg", tableName, columnName, type);
    Table table = Spark3Util.loadIcebergTable(spark, tableName);

    Dataset<Row> dataDF = randomDataDF(table.schema());
    append(tableName, dataDF);

    SparkActions actions = SparkActions.get();
    table.refresh();
    assertThatThrownBy(() -> actions.computeTableEmbeddings(table).columns(columnName).execute())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageStartingWith(
            "Can't compute embeddings on non-string type columns: %s", columnName);
  }

  private GenericRecord createNestedRecord() {
    GenericRecord record = GenericRecord.create(SCHEMA_WITH_NESTED_COLUMN);
    GenericRecord nested = GenericRecord.create(NESTED_STRUCT_TYPE);
    GenericRecord leaf = GenericRecord.create(LEAF_STRUCT_TYPE);
    leaf.set(0, 0L);
    leaf.set(1, "iceberg");
    nested.set(0, leaf);
    record.set(0, nested);
    record.set(1, "data");
    return record;
  }

  private Dataset<Row> randomDataDF(Schema schema) {
    Iterable<InternalRow> rows = RandomData.generateSpark(schema, 10, 0);
    JavaRDD<InternalRow> rowRDD = sparkContext.parallelize(Lists.newArrayList(rows));
    StructType rowSparkType = SparkSchemaUtil.convert(schema);
    return spark.internalCreateDataFrame(JavaRDD.toRDD(rowRDD), rowSparkType, false);
  }

  private void append(String table, Dataset<Row> df) throws NoSuchTableException {
    // fanout writes are enabled as write-time clustering is not supported without Spark extensions
    df.coalesce(1).writeTo(table).option(SparkWriteOptions.FANOUT_ENABLED, "true").append();
  }

  @AfterEach
  public void removeTable() {
    sql("DROP TABLE IF EXISTS %s", tableName);
  }
}
