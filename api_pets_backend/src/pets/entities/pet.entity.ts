import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn } from "typeorm"
import { PetsType } from "./enums"
import { IsIn, Max, Min } from "class-validator"
import { Users } from "src/users/entities/users.entity"

@Entity()

export class Pets {

    @PrimaryGeneratedColumn()
    id: number

    @Column()
    name: string

    @Column()
    color: string

    @Column()
    height : number

    @Column({
        type: 'enum',
        enum: PetsType,
        default: PetsType.DOG,
      })
    type: PetsType

    @Column({ type: 'smallint', unsigned: true, nullable : true })
    @Min(1)
    @Max(4)
    @IsIn([1, 2, 3, 4])
    rating: number;

    @ManyToOne(() => Users, user => user.pets, { nullable: false, onDelete: 'RESTRICT' })
    @JoinColumn({ name: 'user_id' })
    user: Users;

    @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
    created_at: string  

    @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
    updated_at: string  


} 