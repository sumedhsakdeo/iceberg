/*
 *
 *  * Licensed to the Apache Software Foundation (ASF) under one
 *  * or more contributor license agreements.  See the NOTICE file
 *  * distributed with this work for additional information
 *  * regarding copyright ownership.  The ASF licenses this file
 *  * to you under the Apache License, Version 2.0 (the
 *  * "License"); you may not use this file except in compliance
 *  * with the License.  You may obtain a copy of the License at
 *  *
 *  *   http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing,
 *  * software distributed under the License is distributed on an
 *  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  * KIND, either express or implied.  See the License for the
 *  * specific language governing permissions and limitations
 *  * under the License.
 *
 */

package org.apache.spark.sql.index

import dev.langchain4j.data.document.Metadata
import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.output.Response
import org.apache.iceberg.spark.actions.{TextEmbedding, TextEmbeddingBuffer}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.aggregate.{ImperativeAggregate, TypedImperativeAggregate}
import org.apache.spark.sql.catalyst.trees.UnaryLike
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{BinaryType, DataType}
import org.apache.spark.unsafe.types.UTF8String

import java.time.Duration
import java.util.Collections

case class EmbeddingAgg(
                         child: Expression,
                         mutableAggBufferOffset: Int = 0,
                         inputAggBufferOffset: Int = 0) extends TypedImperativeAggregate[TextEmbeddingBuffer]
  with UnaryLike[Expression] {

  override def nullable: Boolean = false

  override def dataType: DataType = BinaryType

  def this(colName: String) = {
    this(col(colName).expr, 0, 0)
  }

  override def createAggregationBuffer(): TextEmbeddingBuffer = {
    val emptyList: java.util.List[TextEmbedding] = Collections.emptyList()
    new TextEmbeddingBuffer(emptyList)
  }

  override def update(buffer: TextEmbeddingBuffer, input: InternalRow): TextEmbeddingBuffer = {
    val value = child.eval(input)
    if (value != null && value.isInstanceOf[UTF8String]) {

      System.setProperty("ai.djl.offline", "true")
      System.setProperty("DJL_OFFLINE", "true")

      val embeddingModel = new AllMiniLmL6V2EmbeddingModel()
      val response : Response[Embedding] = embeddingModel.embed(value.toString)
      print(response.content())

      val ollamaModel : EmbeddingModel = new OllamaEmbeddingModel(
        "http://localhost:11434/",
        "llama3.1",
        Duration.ofSeconds(60),
        3,
        true,
        true,
        Collections.emptyMap()
      )

      val metadata: Metadata = new Metadata()
      val textSegment: TextSegment = TextSegment.from(value.toString, metadata)
      val embedding: Embedding = ollamaModel.embed(textSegment).content()
      val singletonList: java.util.List[TextEmbedding] = Collections.singletonList(new TextEmbedding(textSegment, embedding))
      buffer.merge(new TextEmbeddingBuffer(singletonList))
    } else {
      val emptyList: java.util.List[TextEmbedding] = Collections.emptyList()
      new TextEmbeddingBuffer(emptyList)
    }
  }

  override def merge(buffer: TextEmbeddingBuffer, input: TextEmbeddingBuffer):
  TextEmbeddingBuffer = buffer.merge(input)

  override def eval(buffer: TextEmbeddingBuffer): Any = {
    buffer.serialize()
  }

  override def serialize(buffer: TextEmbeddingBuffer): Array[Byte] = {
    buffer.serialize()
  }

  override def deserialize(storageFormat: Array[Byte]): TextEmbeddingBuffer = {
    TextEmbeddingBuffer.deserialize(storageFormat)
  }

  override def withNewMutableAggBufferOffset(newMutableAggBufferOffset: Int): ImperativeAggregate = {
    copy(mutableAggBufferOffset = newMutableAggBufferOffset)
  }

  override def withNewInputAggBufferOffset(newInputAggBufferOffset: Int): ImperativeAggregate = {
    copy(inputAggBufferOffset = newInputAggBufferOffset)
  }

  override protected def withNewChildInternal(newChild: Expression): Expression = {
    copy(child = newChild)
  }


}


