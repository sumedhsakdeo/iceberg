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

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import dev.langchain4j.data.document.Metadata
import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.aggregate.{ImperativeAggregate, TypedImperativeAggregate}
import org.apache.spark.sql.catalyst.trees.UnaryLike
import org.apache.spark.sql.types.{BinaryType, DataType}

import java.nio.charset.StandardCharsets

case class EmbeddingAgg(
                         child: Expression,
                         mutableAggBufferOffset: Int = 0,
                         inputAggBufferOffset: Int = 0) extends TypedImperativeAggregate[List[(TextSegment, Embedding)]]
  with UnaryLike[Expression] {

  override def nullable: Boolean = false

  override def dataType: DataType = BinaryType

  override def createAggregationBuffer(): List[(TextSegment, Embedding)] = {
    List.empty[(TextSegment, Embedding)]
  }

  override def update(buffer: List[(TextSegment, Embedding)], input: InternalRow): List[(TextSegment, Embedding)] = {
    val value = child.eval(input)
    if (value != null && value.isInstanceOf[String]) {
      val embeddingModel: EmbeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel()
      val metadata: Metadata = new Metadata()
      val textSegment: TextSegment = TextSegment.from(value.toString, metadata)
      val embedding: Embedding = embeddingModel.embed(textSegment).content()
      (textSegment, embedding) :: buffer
    } else {
      null
    }
  }

  override def merge(buffer: List[(TextSegment, Embedding)], input: List[(TextSegment, Embedding)]):
  List[(TextSegment, Embedding)] = buffer ::: input

  override def eval(buffer: List[(TextSegment, Embedding)]): Any = {
    toBytes(buffer)
  }

  private def toBytes(buffer: List[(TextSegment, Embedding)]) = {
    val gson = new Gson()
    gson.toJson(buffer).getBytes(StandardCharsets.UTF_8)
  }

  override def serialize(buffer: List[(TextSegment, Embedding)]): Array[Byte] = {
    toBytes(buffer)
  }

  override def deserialize(storageFormat: Array[Byte]): List[(TextSegment, Embedding)] = {
    val jsonString = new String(storageFormat, StandardCharsets.UTF_8)
    val listType = new TypeToken[List[(TextSegment, Embedding)]]() {}.getType
    val gson = new Gson()
    gson.fromJson(jsonString, listType)
  }

  override protected def withNewChildInternal(newChild: Expression): Expression = {
    copy(child = newChild)
  }

  override def withNewMutableAggBufferOffset(newMutableAggBufferOffset: Int): ImperativeAggregate = {
    copy(mutableAggBufferOffset = newMutableAggBufferOffset)
  }

  override def withNewInputAggBufferOffset(newInputAggBufferOffset: Int): ImperativeAggregate = {
    copy(inputAggBufferOffset = newInputAggBufferOffset)
  }
}
