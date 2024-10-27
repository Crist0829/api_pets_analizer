import {
  MigrationInterface,
  QueryRunner,
  Table,
  TableForeignKey,
} from 'typeorm'

export class PetImageLikes1729985339560 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.createTable(
      new Table({
        name: 'pet_image_likes',
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
      'pet_image_likes',
      new TableForeignKey({
        columnNames: ['user_id'],
        referencedColumnNames: ['id'],
        referencedTableName: 'users',
        onDelete: 'CASCADE',
      }),
    )

    await queryRunner.createForeignKey(
      'pet_image_likes',
      new TableForeignKey({
        columnNames: ['pet_id'],
        referencedColumnNames: ['id'],
        referencedTableName: 'pets',
        onDelete: 'CASCADE',
      }),
    )
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    queryRunner.dropTable('pet_imagen_likes')
  }
}
