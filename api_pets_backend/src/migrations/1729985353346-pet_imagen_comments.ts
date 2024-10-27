import {
  MigrationInterface,
  QueryRunner,
  Table,
  TableForeignKey,
} from 'typeorm'

export class PetImagenComments1729985353346 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.createTable(
      new Table({
        name: 'pet_imagen_comments',
        columns: [
          {
            name: 'id',
            type: 'int',
            isGenerated: true,
            isPrimary: true,
            generationStrategy: 'increment',
          },
          {
            name: 'pet_id',
            type: 'int',
            isNullable: false,
          },
          {
            name: 'user_id',
            type: 'int',
            isNullable: false,
          },
          {
            name: 'pet_imagen_comment_id',
            type: 'int',
            isNullable: true,
          },
          {
            name: 'created_at',
            type: 'timestamp',
            default: 'now()',
            isNullable: false,
          },
          {
            name: 'updated_at',
            type: 'timestamp',
            default: 'now()',
            isNullable: false,
          },
        ],
      }),
    )

    await queryRunner.createForeignKey(
      'pet_imagen_comments',
      new TableForeignKey({
        columnNames: ['user_id'],
        referencedColumnNames: ['id'],
        referencedTableName: 'users',
        onDelete: 'CASCADE',
      }),
    )

    await queryRunner.createForeignKey(
      'pet_imagen_comments',
      new TableForeignKey({
        columnNames: ['pet_imagen_comment_id'],
        referencedColumnNames: ['id'],
        referencedTableName: 'pet_imagen_comments',
        onDelete: 'CASCADE',
      }),
    )
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    queryRunner.dropTable('pet_imagen_comments')
  }
}
