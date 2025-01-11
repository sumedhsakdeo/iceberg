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

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Snapshot;
import org.apache.iceberg.Table;
import org.apache.iceberg.puffin.Blob;
import org.apache.iceberg.puffin.PuffinCompressionCodec;
import org.apache.iceberg.puffin.StandardBlobTypes;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableMap;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.spark.SparkTableUtil;
import org.apache.iceberg.types.Types;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.index.EmbeddingAgg;
import scala.Predef;
import scala.Tuple2;
import scala.collection.JavaConverters;

public class TextEmbeddingPuffinWriter {

  private static final String NUM_EMBEDDINGS = "num_embeddings";

  private TextEmbeddingPuffinWriter() {}

  static List<Blob> generateBlobs(
      SparkSession spark,
      Table table,
      Snapshot snapshot,
      List<String> columns,
      String modelName,
      Map<String, String> modelInputs) {
    Row embeddings = computeEmbeddings(spark, table, snapshot, columns, modelName, modelInputs);
    Schema schema = table.schemas().get(snapshot.schemaId());
    List<Blob> blobs = Lists.newArrayList();
    for (int i = 0; i < columns.size(); i++) {
      Types.NestedField field = schema.findField(columns.get(i));
      byte[] buffer = (byte[]) embeddings.get(i);
      TextEmbeddingBuffer textEmbeddingBuffer = TextEmbeddingBuffer.deserialize(buffer);
      blobs.add(toBlob(field, buffer, textEmbeddingBuffer.getTextEmbeddings().size(), snapshot));
    }
    return blobs;
  }

  private static Blob toBlob(
      Types.NestedField field, byte[] serializedEmbeddings, long numEmbeddings, Snapshot snapshot) {
    return new Blob(
        StandardBlobTypes.EMBEDDINGS_V1,
        ImmutableList.of(field.fieldId()),
        snapshot.snapshotId(),
        snapshot.sequenceNumber(),
        ByteBuffer.wrap(serializedEmbeddings),
        PuffinCompressionCodec.ZSTD,
        ImmutableMap.of(NUM_EMBEDDINGS, String.valueOf(numEmbeddings)));
  }

  private static Row computeEmbeddings(
      SparkSession spark,
      Table table,
      Snapshot snapshot,
      List<String> colNames,
      String modelName,
      Map<String, String> modelInputs) {
    Dataset<Row> inputDF = SparkTableUtil.loadTable(spark, table, snapshot.snapshotId());
    return inputDF.select(toAggColumns(colNames, modelName, modelInputs)).first();
  }

  private static Column[] toAggColumns(
      List<String> colNames, String modelName, Map<String, String> modelInputs) {
    return colNames.stream()
        .map(column -> TextEmbeddingPuffinWriter.toAggColumn(column, modelName, modelInputs))
        .toArray(Column[]::new);
  }

  private static Column toAggColumn(
      String colName, String modelName, Map<String, String> modelInputs) {
    EmbeddingAgg agg =
        new EmbeddingAgg(
            colName,
            modelName,
            JavaConverters.mapAsScalaMapConverter(modelInputs)
                .asScala()
                .toMap(Predef.<Tuple2<String, String>>conforms()));
    return new Column(agg.toAggregateExpression());
  }
}
