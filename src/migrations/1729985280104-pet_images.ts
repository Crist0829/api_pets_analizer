import {
  MigrationInterface,
  QueryRunner,
  Table,
  TableForeignKey,
} from 'typeorm'

export class PetImages1729985280104 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.createTable(
      new Table({
        name: 'pet_images',
        columns: [
          {
            name: 'id',
            type: 'int',
            isPrimary: true,
            isGenerated: true,
            generationStrategy: 'increment',
          },
          {
            name: 'name',
            type: 'varchar',
            isNullable: false,
          },

          {
            name : 'verified', 
            type : 'tinyint', 
            isNullable : false, 
            default : 0
          },

          {
            name: 'url',
            type: 'varchar',
            isNullable: false,
          },

          {
            name: 'pet_id',
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
      'pet_images',
      new TableForeignKey({
        columnNames: ['pet_id'],
        referencedColumnNames: ['id'],
        referencedTableName: 'pets',
        onDelete: 'CASCADE',
      }),
    )
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    queryRunner.dropTable('pet_images')
  }
}
