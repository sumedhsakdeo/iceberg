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

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;
import org.apache.iceberg.GenericBlobMetadata;
import org.apache.iceberg.GenericStatisticsFile;
import org.apache.iceberg.HasTableOperations;
import org.apache.iceberg.IcebergBuild;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Snapshot;
import org.apache.iceberg.StatisticsFile;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableOperations;
import org.apache.iceberg.actions.ComputeTableEmbeddings;
import org.apache.iceberg.actions.ImmutableComputeTableEmbeddings;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.puffin.Blob;
import org.apache.iceberg.puffin.Puffin;
import org.apache.iceberg.puffin.PuffinWriter;
import org.apache.iceberg.relocated.com.google.common.base.Preconditions;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableSet;
import org.apache.iceberg.spark.JobGroupInfo;
import org.apache.iceberg.types.Types;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ComputeTableEmbeddingsSparkAction
    extends BaseSparkAction<ComputeTableEmbeddingsSparkAction> implements ComputeTableEmbeddings {

  private static final Logger LOG =
      LoggerFactory.getLogger(ComputeTableEmbeddingsSparkAction.class);
  private static final ComputeTableEmbeddings.Result EMPTY_RESULT =
      ImmutableComputeTableEmbeddings.Result.builder().build();

  private final Table table;
  private List<String> columns;
  private Snapshot snapshot;
  private String modelName;
  private Map<String, String> modelInputs;

  ComputeTableEmbeddingsSparkAction(SparkSession spark, Table table) {
    super(spark);
    this.table = table;
    this.snapshot = table.currentSnapshot();
  }

  @Override
  protected ComputeTableEmbeddingsSparkAction self() {
    return this;
  }

  @Override
  public ComputeTableEmbeddings columns(String... newColumns) {
    Preconditions.checkArgument(
        newColumns != null && newColumns.length > 0, "Columns cannot be null/empty");
    this.columns = ImmutableList.copyOf(ImmutableSet.copyOf(newColumns));
    return this;
  }

  @Override
  public ComputeTableEmbeddings snapshot(long newSnapshotId) {
    Snapshot newSnapshot = table.snapshot(newSnapshotId);
    Preconditions.checkArgument(newSnapshot != null, "Snapshot not found: %s", newSnapshotId);
    this.snapshot = newSnapshot;
    return this;
  }

  @Override
  public ComputeTableEmbeddings modelName(String newModelName) {
    Preconditions.checkArgument(newModelName != null && !newModelName.isEmpty(), "Model name cannot be null or empty");
    this.modelName = newModelName;
    return this;
  }

  @Override
  public ComputeTableEmbeddings modelInputs(Map<String, String> newModelInputs) {
    this.modelInputs = newModelInputs;
    return this;
  }

  @Override
  public Result execute() {
    if (snapshot == null) {
      LOG.info("No snapshot to compute embeddings for table {}", table.name());
      return EMPTY_RESULT;
    }
    validateColumns();
    JobGroupInfo info = newJobGroupInfo("COMPUTE-TABLE-EMBEDDINGS", jobDesc());
    return withJobGroupInfo(info, this::doExecute);
  }

  private ComputeTableEmbeddings.Result doExecute() {
    LOG.info(
        "Computing embedding for columns {} in {} (snapshot {})",
        columns(),
        table.name(),
        snapshotId());
    List<Blob> blobs = generateEmbeddings();
    StatisticsFile statisticsFile = writeStatsFile(blobs);
    table.updateStatistics().setStatistics(snapshotId(), statisticsFile).commit();
    return ImmutableComputeTableEmbeddings.Result.builder().statisticsFile(statisticsFile).build();
  }

  private StatisticsFile writeStatsFile(List<Blob> blobs) {
    LOG.info("Writing embeddings for table {} for snapshot {}", table.name(), snapshotId());
    OutputFile outputFile = table.io().newOutputFile(outputPath());
    try (PuffinWriter writer = Puffin.write(outputFile).createdBy(appIdentifier()).build()) {
      blobs.forEach(writer::add);
      writer.finish();
      return new GenericStatisticsFile(
          snapshotId(),
          outputFile.location(),
          writer.fileSize(),
          writer.footerSize(),
          GenericBlobMetadata.from(writer.writtenBlobsMetadata()));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private List<Blob> generateEmbeddings() {
    return TextEmbeddingUtil.generateBlobs(
        spark(), table, snapshot, columns, modelName, modelInputs);
  }

  private void validateColumns() {
    Schema schema = table.schemas().get(snapshot.schemaId());
    Preconditions.checkArgument(!columns().isEmpty(), "No columns found to compute embeddings");
    for (String columnName : columns()) {
      Types.NestedField field = schema.findField(columnName);
      Preconditions.checkArgument(field != null, "Can't find column %s in %s", columnName, schema);
      Preconditions.checkArgument(
          field.type().isPrimitiveType(),
          "Can't compute embeddings on non-primitive type column: %s (%s)",
          columnName,
          field.type());
      Preconditions.checkArgument(
          field.type().typeId() == Types.StringType.get().typeId(),
          "Can't compute embeddings on non-string type columns: %s (%s)",
          columnName,
          field.type());
    }
  }

  private List<String> columns() {
    if (columns == null) {
      Schema schema = table.schemas().get(snapshot.schemaId());
      this.columns =
          schema.columns().stream()
              .filter(nestedField -> nestedField.type().isPrimitiveType())
              .map(Types.NestedField::name)
              .collect(Collectors.toList());
    }
    return columns;
  }

  private String appIdentifier() {
    String icebergVersion = IcebergBuild.fullVersion();
    String sparkVersion = spark().version();
    return String.format("Iceberg %s Spark %s", icebergVersion, sparkVersion);
  }

  private long snapshotId() {
    return snapshot.snapshotId();
  }

  private String jobDesc() {
    return String.format(
        "Computing table embedding for %s (snapshot_id=%s, columns=%s)",
        table.name(), snapshotId(), columns());
  }

  private String outputPath() {
    TableOperations operations = ((HasTableOperations) table).operations();
    String fileName = String.format("%s-%s.embedding", snapshotId(), UUID.randomUUID());
    return operations.metadataFileLocation(fileName);
  }
}
